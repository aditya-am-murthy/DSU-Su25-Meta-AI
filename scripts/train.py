#!/usr/bin/env python3
"""
Main training script for the Video Similarity Learning project.
This script trains models with wandb logging and experiment tracking.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import VideoDataset, VideoPairDataset, create_dataloader
from utils.model_utils import create_model, save_model, load_model
from utils.training_utils import (
    setup_wandb, create_optimizer, create_scheduler,
    ContrastiveLoss, TripletLoss, train_model
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Video Similarity Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/videos',
                       help='Directory containing video data')
    parser.add_argument('--metadata_file', type=str, default='sample_metadata.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--pairs_file', type=str, default='similarity_pairs.csv',
                       help='Path to similarity pairs CSV file')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='similarity',
                       choices=['similarity', 'classification'],
                       help='Type of model to train')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'resnet101'],
                       help='Backbone architecture')
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='Feature dimension')
    parser.add_argument('--num_frames', type=int, default=30,
                       help='Number of frames per video')
    parser.add_argument('--pooling_method', type=str, default='mean',
                       choices=['mean', 'max', 'attention'],
                       help='Temporal pooling method')
    parser.add_argument('--similarity_method', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'learned'],
                       help='Similarity computation method')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes (for classification)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--loss_margin', type=float, default=1.0,
                       help='Margin for contrastive/triplet loss')
    
    # Data loading arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Logging and saving arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Frequency to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='video-similarity-learning',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_experiment', type=str, default=None,
                       help='Weights & Biases experiment name')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_config(config_file):
    """Load configuration from YAML file."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return {}


def setup_device(device_arg):
    """Setup device for training."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_datasets(args, data_dir):
    """Create training and validation datasets."""
    print("Creating datasets...")
    
    # Define dataset paths
    metadata_file = os.path.join(data_dir, args.metadata_file)
    pairs_file = os.path.join(data_dir, args.pairs_file)
    
    if args.model_type == 'similarity':
        # Create similarity learning datasets
        train_dataset = VideoPairDataset(
            data_dir=data_dir,
            metadata_file=pairs_file,
            max_frames=args.num_frames,
            image_size=args.image_size,
            transform=None
        )
        
        # For validation, we'll use the same dataset but with different transforms
        val_dataset = VideoPairDataset(
            data_dir=data_dir,
            metadata_file=pairs_file,
            max_frames=args.num_frames,
            image_size=args.image_size,
            transform=None
        )
        
    else:  # classification
        # Create classification datasets
        train_dataset = VideoDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            max_frames=args.num_frames,
            image_size=args.image_size,
            is_training=True,
            transform=None
        )
        
        val_dataset = VideoDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            max_frames=args.num_frames,
            image_size=args.image_size,
            is_training=False,
            transform=None
        )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset


def create_loss_function(args, model_type):
    """Create loss function based on model type."""
    if model_type == 'similarity':
        # For similarity learning, use contrastive loss
        criterion = ContrastiveLoss(margin=args.loss_margin)
        print(f"Using ContrastiveLoss with margin {args.loss_margin}")
    else:
        # For classification, use cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    
    return criterion


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration file if provided
    config = load_config(args.config_file)
    
    # Override args with config values
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Create data directory path
    data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please run the data download script first: python scripts/download_data.py")
        return
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args, data_dir)
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=args.model_type,
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        num_frames=args.num_frames,
        pooling_method=args.pooling_method,
        similarity_method=args.similarity_method,
        num_classes=args.num_classes
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    criterion = create_loss_function(args, args.model_type)
    criterion = criterion.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        step_size=10,
        gamma=0.1
    )
    
    # Setup Weights & Biases
    config_dict = vars(args)
    setup_wandb(
        project_name=args.wandb_project,
        experiment_name=args.wandb_experiment,
        config=config_dict
    )
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(project_root, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("="*60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.num_epochs,
        model_type=args.model_type,
        checkpoint_dir=checkpoint_dir,
        save_freq=args.save_freq
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    save_model(model, final_model_path, optimizer, args.num_epochs, history['val_loss'][-1])
    
    # Log final metrics
    wandb.log({
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'best_val_loss': min(history['val_loss'])
    })
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {final_model_path}")
    
    # Close wandb
    wandb.finish()


if __name__ == "__main__":
    main() 