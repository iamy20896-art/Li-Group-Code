import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
from model.earthformer.utilss.optim import SequentialLR, warmup_lambda
from model.earthformer.utilss.utils import get_parameter_names
from model.Earthformer_Adapter import Earthformer
from data.moving_mnist import get_dataloaders
from utils.metrics import calculate_mae, calculate_mse


def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(train_loader, desc='train'):
        inputs = inputs.float().to(device) / 255.0
        targets = targets.float().to(device) / 255.0

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_mae = 0
    total_mse = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='val'):
            inputs = inputs.float().to(device) / 255.0
            targets = targets.float().to(device) / 255.0

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_mae += calculate_mae(outputs, targets)
            total_mse += calculate_mse(outputs, targets)

    avg_loss = total_loss / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    avg_mse = total_mse / len(test_loader)

    return avg_loss, avg_mae, avg_mse


def train_model(model, train_loader, test_loader, criterion, optimizer, device, args, scheduler=None):
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_mse': [], 'lr': []}

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}, Testing samples: {len(test_loader.dataset)}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Number of epochs: {args.epochs}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        history['train_loss'].append(train_loss)

        val_loss, val_mae, val_mse = validate(model, test_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)

        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            history['lr'].append(current_lr)
            print(f'Current LR: {current_lr:.8f}')
        else:
            history['lr'].append(args.lr)

        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, MSE: {val_mse:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'Saved best model (Val Loss: {val_loss:.6f})')

    torch.save(history, 'results/history.pth')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history['val_mae'])
    axes[1].set_title('MAE')
    axes[1].grid()

    axes[2].plot(history['val_mse'])
    axes[2].set_title('MSE')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.close()


def visualize_predictions(model, test_loader, device, num_samples=2):
    """可视化预测结果"""
    model.eval()

    inputs, targets = next(iter(test_loader))
    inputs = inputs.float().to(device) / 255.0
    targets = targets.float().to(device) / 255.0

    with torch.no_grad():
        outputs = model(inputs)

    mae = calculate_mae(outputs, targets)
    mse = calculate_mse(outputs, targets)

    print(f'MAE: {mae:.6f}, MSE: {mse:.6f}')

    inputs = inputs.cpu()
    targets = targets.cpu()
    outputs = outputs.cpu()

    os.makedirs('results', exist_ok=True)

    num_samples = min(num_samples, inputs.shape[0])
    num_input_frames = inputs.shape[1]
    num_output_frames = targets.shape[1]
    total_cols = num_input_frames + num_output_frames

    fig, axes = plt.subplots(num_samples * 2, total_cols,
                             figsize=(total_cols * 0.8, num_samples * 2 * 0.8))

    if num_samples == 1:
        axes = axes.reshape(2, -1)

    for sample_idx in range(num_samples):
        row_truth = sample_idx * 2
        row_pred = sample_idx * 2 + 1

        col = 0

        for t in range(num_input_frames):
            axes[row_truth, col].imshow(inputs[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_truth, col].axis('off')
            if sample_idx == 0:
                axes[row_truth, col].set_title(f'T={t + 1}', fontsize=9)
            col += 1

        for t in range(num_output_frames):
            axes[row_truth, col].imshow(targets[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_truth, col].axis('off')
            if sample_idx == 0:
                axes[row_truth, col].set_title(f'T={num_input_frames + t + 1}', fontsize=9)
            col += 1

        col = 0

        for i in range(num_input_frames):
            axes[row_pred, col].axis('off')
            col += 1

        for t in range(num_output_frames):
            axes[row_pred, col].imshow(outputs[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_pred, col].axis('off')
            col += 1

        fig.text(0.02, 1 - (sample_idx * 2 + 0.5) / (num_samples * 2),
                 'Truth', ha='right', va='center', fontsize=10, weight='bold')
        fig.text(0.02, 1 - (sample_idx * 2 + 1.5) / (num_samples * 2),
                 'Pred', ha='right', va='center', fontsize=10, weight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def configure_optimizer_and_scheduler(model, args, train_loader):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]

    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if n in decay_parameters],
        'weight_decay': args.weight_decay
    }, {
        'params': [p for n, p in model.named_parameters() if n not in decay_parameters],
        'weight_decay': 0.0
    }]

    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_iter = int(np.round(args.warmup_percentage * total_steps))

    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=warmup_lambda(
            warmup_steps=warmup_iter,
            min_lr_ratio=args.warmup_min_lr_ratio
        )
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_steps - warmup_iter),
        eta_min=args.min_lr_ratio * args.lr
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iter]
    )

    print(f"\n优化器配置:")
    print(f"  优化器: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  学习率调度: Warmup ({warmup_iter} steps) + Cosine Annealing")
    print(f"  总步数: {total_steps}, Warmup步数: {warmup_iter}")
    print(f"  最低学习率: {args.min_lr_ratio * args.lr:.8f}\n")

    return optimizer, lr_scheduler


def main():
    parser = argparse.ArgumentParser(description='Moving MNIST with Earthformer')

    parser.add_argument('--visualize-only', action='store_true', default=False,
                        help='only visualize predictions using a pre-trained model')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8, 原实现micro_batch_size=8)')

    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs (default: 100, cfg.yaml中是100)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3, cfg.yaml中是0.001)')

    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--no-visualize', action='store_true', default=False,
                        help='do not visualize after training')

    parser.add_argument('--num-samples', type=int, default=2,
                        help='number of samples to visualize (default: 2)')

    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for optimizer (default: 1e-5)')

    parser.add_argument('--warmup-percentage', type=float, default=0.2,
                        help='Warmup steps percentage (default: 0.2)')

    parser.add_argument('--warmup-min-lr-ratio', type=float, default=0.0,
                        help='Minimum LR during warmup as ratio of initial LR (default: 0.0)')

    parser.add_argument('--min-lr-ratio', type=float, default=1e-3,
                        help='Minimum LR as ratio of initial LR (default: 1e-3)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ========== create model ==========
    model = Earthformer(input_channel=10, num_classes=10).to(device)
    # ========================================================

    if args.visualize_only:
        # only visualize mode - load trained model
        if not os.path.exists('checkpoints/best_model.pth'):
            print("error: 'checkpoints/best_model.pth' not found")
            return

        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
        _, test_loader = get_dataloaders(batch_size=max(4, args.num_samples), num_workers=0)
        visualize_predictions(model, test_loader, device, num_samples=args.num_samples)

    else:
        # training mode
        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

        criterion = nn.MSELoss()

        optimizer, lr_scheduler = configure_optimizer_and_scheduler(
            model, args, train_loader
        )

        train_model(model, train_loader, test_loader, criterion, optimizer, device, args, lr_scheduler)

        if not args.no_visualize:
            _, test_loader = get_dataloaders(batch_size=max(4, args.num_samples), num_workers=args.workers)
            visualize_predictions(model, test_loader, device, num_samples=args.num_samples)


if __name__ == '__main__':
    main()