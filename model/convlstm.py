"""
paper : Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting"
https://arxiv.org/abs/1506.04214

code : https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py

"""

import torch
from torch import nn
from collections import OrderedDict


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, num_filter, height, width, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channel + num_filter,
            out_channels=num_filter * 4,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # peephole 门控参数
        self.Wci = nn.Parameter(torch.zeros(1, num_filter, height, width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_filter, height, width))
        self.Wco = nn.Parameter(torch.zeros(1, num_filter, height, width))

        self.input_channel = input_channel
        self.num_filter = num_filter
        self.height = height
        self.width = width

    def forward(self, inputs=None, states=None, seq_len=10):
        """
        inputs: [T, B, C, H, W]
        states: (h, c)
        """
        device = next(self.parameters()).device

        if states is None:
            h = torch.zeros(inputs.size(1), self.num_filter, self.height, self.width, device=device)
            c = torch.zeros(inputs.size(1), self.num_filter, self.height, self.width, device=device)
        else:
            h, c = states

        outputs = []
        for t in range(seq_len):
            if inputs is None:
                x = torch.zeros(h.size(0), self.input_channel, self.height, self.width, device=device)
            else:
                x = inputs[t]

            combined = torch.cat([x, h], dim=1)
            conv_out = self.conv(combined)
            i, f, tmp_c, o = torch.chunk(conv_out, 4, dim=1)

            i = torch.sigmoid(i + self.Wci * c)
            f = torch.sigmoid(f + self.Wcf * c)
            c = f * c + i * torch.tanh(tmp_c)
            o = torch.sigmoid(o + self.Wco * c)
            h = o * torch.tanh(c)
            outputs.append(h)

        return torch.stack(outputs), (h, c)


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(
                in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(
                in_channels=v[0], out_channels=v[1],
                kernel_size=v[2], stride=v[3], padding=v[4]
            )
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, f"stage{index}", make_layers(params))
            setattr(self, f"rnn{index}", rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = input.reshape(-1, input_channel, height, width)
        input = subnet(input)
        input = input.reshape(seq_number, batch_size, input.size(1), input.size(2), input.size(3))
        outputs_stage, state_stage = rnn(input, None, seq_len=seq_number)
        return outputs_stage, state_stage

    def forward(self, input):
        hidden_states = []
        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(
                input, getattr(self, f"stage{i}"), getattr(self, f"rnn{i}")
            )
            hidden_states.append(state_stage)
        return tuple(hidden_states)


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, out_len=10):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)
        self.out_len = out_len

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, f"rnn{self.blocks - index}", rnn)
            setattr(self, f"stage{self.blocks - index}", make_layers(params))

    def forward_by_stage(self, input, state, subnet, rnn):
        input, state_stage = rnn(input, state, seq_len=self.out_len)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = input.reshape(-1, input_channel, height, width)
        input = subnet(input)
        input = input.reshape(seq_number, batch_size, input.size(1), input.size(2), input.size(3))
        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(
            None, hidden_states[-1],
            getattr(self, f"stage{self.blocks}"),
            getattr(self, f"rnn{self.blocks}")
        )
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(
                input, hidden_states[i - 1],
                getattr(self, f"stage{i}"), getattr(self, f"rnn{i}")
            )
        return input


class ConvLSTM(nn.Module):
    def __init__(self, height=64, width=64, input_channel=1, output_channel=1,
                 hidden_channels=[8, 16], out_len=10):
        super().__init__()

        # 定义 encoder 子网络与 RNN 单元
        encoder_subnets = [
            OrderedDict({
                'conv1_leaky': [input_channel, 8, 3, 1, 1],
                'pool1': [2, 2, 0],
            }),
            OrderedDict({
                'conv2_leaky': [8, 16, 3, 1, 1],
                'pool2': [2, 2, 0],
            }),
        ]
        encoder_rnns = [
            ConvLSTMCell(8, hidden_channels[0], height // 2, width // 2),
            ConvLSTMCell(16, hidden_channels[1], height // 4, width // 4),
        ]

        # 定义 decoder（forecaster）子网络与 RNN
        forecaster_subnets = [
            OrderedDict({
                'deconv1_leaky': [hidden_channels[1], 8, 4, 2, 1],
            }),
            OrderedDict({
                'deconv2_leaky': [8, output_channel, 4, 2, 1],
            }),
        ]
        forecaster_rnns = [
            ConvLSTMCell(hidden_channels[1], hidden_channels[1], height // 4, width // 4),
            ConvLSTMCell(8, hidden_channels[0], height // 2, width // 2),
        ]

        self.encoder = Encoder(encoder_subnets, encoder_rnns)
        self.forecaster = Forecaster(forecaster_subnets, forecaster_rnns, out_len=out_len)

    def forward(self, x):
        """
        输入:  [B, 10, H, W]
        输出:  [B, 10, H, W]
        """
        # 调整输入为 [T, B, C, H, W]
        x = x.unsqueeze(2).permute(1, 0, 2, 3, 4)
        hidden_states = self.encoder(x)
        output = self.forecaster(hidden_states)  # [T, B, C, H, W]
        output = output.permute(1, 0, 2, 3, 4).squeeze(2)  # [B, 10, H, W]
        return output


if __name__ == "__main__":
    model = ConvLSTM(height=64, width=64, input_channel=1, output_channel=1)
    x = torch.randn(2, 10, 64, 64)
    y = model(x)
    print("✅ 输出形状:", y.shape)  # 应为 [2, 10, 64, 64]

# import torch
# from torch import nn

# class ConvLSTMCell(nn.Module):
#     def __init__(self, input_channel, num_filter, height, width, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.conv = nn.Conv2d(
#             in_channels=input_channel + num_filter,
#             out_channels=num_filter * 4,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )

#         # peephole 门控参数
#         self.Wci = nn.Parameter(torch.zeros(1, num_filter, height, width))
#         self.Wcf = nn.Parameter(torch.zeros(1, num_filter, height, width))
#         self.Wco = nn.Parameter(torch.zeros(1, num_filter, height, width))

#         self.input_channel = input_channel
#         self.num_filter = num_filter
#         self.height = height
#         self.width = width

#     def forward(self, inputs=None, states=None, seq_len=10):
#         """
#         inputs: [T, B, C, H, W]
#         states: (h, c)
#         """
#         device = next(self.parameters()).device

#         if states is None:
#             h = torch.zeros(inputs.size(1), self.num_filter, self.height, self.width, device=device)
#             c = torch.zeros(inputs.size(1), self.num_filter, self.height, self.width, device=device)
#         else:
#             h, c = states

#         outputs = []
#         for t in range(seq_len):
#             if inputs is None:
#                 x = torch.zeros(h.size(0), self.input_channel, self.height, self.width, device=device)
#             else:
#                 x = inputs[t]

#             combined = torch.cat([x, h], dim=1)
#             conv_out = self.conv(combined)
#             i, f, tmp_c, o = torch.chunk(conv_out, 4, dim=1)

#             i = torch.sigmoid(i + self.Wci * c)
#             f = torch.sigmoid(f + self.Wcf * c)
#             c = f * c + i * torch.tanh(tmp_c)
#             o = torch.sigmoid(o + self.Wco * c)
#             h = o * torch.tanh(c)

#             outputs.append(h)

#         return torch.stack(outputs), (h, c)

# class ConvLSTM(nn.Module):
#     def __init__(self, input_channel=10, output_channel=10, hidden_dim=64, height=64, width=64):
#         super().__init__()
#         """
#         Li Group AI Code Compatible ConvLSTM Model
#         输入: [B, 10, H, W]
#         输出: [B, 10, H, W]
#         """
#         self.height = height
#         self.width = width
#         self.hidden_dim = hidden_dim

#         # 单层 ConvLSTM（带 Peephole）
#         self.convlstm = ConvLSTMCell(
#             input_channel=1, num_filter=hidden_dim,
#             height=height, width=width, kernel_size=3
#         )

#         # 输出映射
#         self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

#         # 预测未来帧长度
#         self.future_steps = output_channel

#     def forward(self, x):
#         """
#         x: [B, 10, H, W]
#         return: [B, 10, H, W]
#         """
#         B, T, H, W = x.shape
#         # [T, B, C, H, W]
#         x = x.unsqueeze(2).permute(1, 0, 2, 3, 4)  # [T, B, 1, H, W]

#         outputs, (h, c) = self.convlstm(x, None, seq_len=T)

#         preds = []
#         frame = outputs[-1]  # 最后时刻的隐藏状态
#         for _ in range(self.future_steps):
#             frame, (h, c) = self.convlstm(None, (h, c), seq_len=1)
#             pred = self.head(frame[-1])
#             preds.append(pred)

#         preds = torch.stack(preds, dim=1).squeeze(2)  # [B, 10, H, W]
#         return preds
