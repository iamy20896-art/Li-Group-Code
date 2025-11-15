"""
paper: Earthformer: Exploring Space-Time Transformers for Earth System Forecasting

code: https://github.com/amazon-science/earth-forecasting-transformer
"""

import torch
import torch.nn as nn

from model.earthformer.cuboid_transformer import CuboidTransformerModel


class EarthformerAdapter(nn.Module):
    """
    将 Earthformer 适配到标准接口

    模型接口规范：
    - 输入: [Batch, 10, Height, Width] - 10帧灰度图像
    - 输出: [Batch, 10, Height, Width] - 后续10帧灰度图像

    内部实现：
    - Earthformer内部使用格式: (B, T, H, W, C)
    - 会自动进行格式转换，用户只需按照标准接口使用即可
    """

    def __init__(self,
                 input_channel=10,  # 输入帧数
                 num_classes=10,  # 输出帧数
                 height=64,
                 width=64,
                 base_units=64,
                 enc_depth=[4, 4],
                 dec_depth=[4, 4],
                 num_heads=4):
        """
        Parameters
        ----------
        input_channel: int
            输入帧数（输入序列长度）
        num_classes: int
            输出帧数（输出序列长度，注意：这里的命名是为了兼容原接口）
        height: int
            图像高度
        width: int
            图像宽度
        base_units: int
            Earthformer 的基础单元数
        enc_depth: list
            编码器每层深度
        dec_depth: list
            解码器每层深度
        num_heads: int
            注意力头数
        """
        super(EarthformerAdapter, self).__init__()

        self.input_channel = input_channel
        self.num_classes = num_classes
        self.height = height
        self.width = width

        self.model = CuboidTransformerModel(
            input_shape=(input_channel, height, width, 1),  # (T, H, W, C)
            target_shape=(num_classes, height, width, 1),
            base_units=base_units,
            block_units=None,
            scale_alpha=1.0,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            num_heads=num_heads,
            attn_drop=0.1,
            proj_drop=0.1,
            ffn_drop=0.1,
            downsample=2,
            downsample_type='patch_merge',
            upsample_type="upsample",
            upsample_kernel_size=3,
            dec_cross_start=0,
            enc_use_inter_ffn=True,
            enc_attn_patterns=['axial'] * len(enc_depth),
            dec_use_inter_ffn=True,
            dec_hierarchical_pos_embed=False,
            dec_self_attn_patterns=['axial'] * len(dec_depth),
            dec_cross_attn_patterns=['cross_1x1'] * len(dec_depth),
            dec_cross_last_n_frames=None,
            dec_use_first_self_attn=False,
            num_global_vectors=0,
            use_dec_self_global=False,
            dec_self_update_global=True,
            use_dec_cross_global=False,
            use_global_vector_ffn=False,
            use_global_self_attn=False,
            separate_global_qkv=False,
            global_dim_ratio=1,
            initial_downsample_type="conv",
            initial_downsample_activation="leaky",
            initial_downsample_scale=2,
            initial_downsample_conv_layers=2,
            final_upsample_conv_layers=1,
            ffn_activation='gelu',
            gated_ffn=False,
            norm_layer='layer_norm',
            padding_type='zeros',
            pos_embed_type='t+hw',
            z_init_method='zeros',
            checkpoint_level=2,
            use_relative_pos=True,
            self_attn_use_final_proj=True,
            attn_linear_init_mode="0",
            ffn_linear_init_mode="0",
            conv_init_mode="0",
            down_up_linear_init_mode="0",
            norm_init_mode="0",
        )

    def forward(self, x):
        """
        前向传播

        接口规范：
        - 输入: [Batch, 10, Height, Width] - 10帧灰度图像
        - 输出: [Batch, 10, Height, Width] - 后续10帧灰度图像

        Parameters
        ----------
        x: torch.Tensor
            输入，shape: [Batch, 10, Height, Width]
            其中 10 是输入帧数（input_channel），Height 和 Width 是图像尺寸

        Returns
        -------
        output: torch.Tensor
            输出，shape: [Batch, 10, Height, Width]
            其中 10 是输出帧数（num_classes），Height 和 Width 与输入相同
        """

        if len(x.shape) == 4:
            x = x.unsqueeze(-1)  # [B, T, H, W] -> [B, T, H, W, 1]

        output = self.model(x)  # 输出: (B, T_out, H, W, C)

        output = output.squeeze(-1)  # [B, T, H, W, 1] -> [B, T, H, W]

        return output


def Earthformer(input_channel=10, num_classes=10, height=64, width=64, **kwargs):
    """
    创建 Earthformer 模型（便捷函数）

    参数与原项目保持兼容

    Parameters
    ----------
    input_channel: int
        输入帧数
    num_classes: int
        输出帧数
    height: int
        图像高度
    width: int
        图像宽度
    **kwargs
        其他 EarthformerAdapter 参数

    Returns
    -------
    model: EarthformerAdapter
        Earthformer 模型实例
    """
    return EarthformerAdapter(
        input_channel=input_channel,
        num_classes=num_classes,
        height=height,
        width=width,
        **kwargs
    )


