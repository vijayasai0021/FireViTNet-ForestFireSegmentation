import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from models.firevitnet import FireViTNet
from utils.dataset import FireDataset
from utils.loss import CombinedLoss
from utils.metrics import calculate_metrics
from training.lightning_module import FireViTNetLightning

def main():
    parser = argparse.ArgumentParser(description='Train FireViTNet')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = FireDataset(
        data_dir=Path(args.data_dir) / 'train',
        input_size=(args.input_size, args.input_size),
        augment=True
    )
    
    val_dataset = FireDataset(
        data_dir=Path(args.data_dir) / 'val',
        input_size=(args.input_size, args.input_size),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = FireViTNetLightning(
        num_classes=2,
        learning_rate=args.learning_rate
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        precision=16,  # Mixed precision training
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename='firevitnet-{epoch:02d}-{val_f1:.4f}',
                monitor='val_f1',
                mode='max',
                save_top_k=3
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_f1',
                patience=10,
                mode='max'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ],
        logger=pl.loggers.TensorBoardLogger('logs', name='firevitnet')
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")

if __name__ == '__main__':
    main()
