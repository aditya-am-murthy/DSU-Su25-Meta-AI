"""
Training utilities for the Video Similarity Learning project.
This module provides loss functions, metrics, and training utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import wandb
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for similarity learning.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, similarity: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            similarity: Similarity scores (batch_size, 1)
            labels: Ground truth labels (1 for similar, 0 for different)
            
        Returns:
            Loss value
        """
        # Convert similarity to distance (assuming similarity is cosine similarity)
        distance = 1 - similarity.squeeze()
        
        # Contrastive loss: similar pairs should have small distance, different pairs should have large distance
        loss_similar = labels * torch.pow(distance, 2)  # For similar pairs
        loss_different = (1 - labels) * torch.pow(torch.clamp(self.margin - distance, min=0), 2)  # For different pairs
        
        loss = torch.mean(loss_similar + loss_different)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for similarity learning.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor features
            positive: Positive features (same class as anchor)
            negative: Negative features (different class from anchor)
            
        Returns:
            Loss value
        """
        # Compute distances
        pos_dist = torch.sum(torch.pow(anchor - positive, 2), dim=1)
        neg_dist = torch.sum(torch.pow(anchor - negative, 2), dim=1)
        
        # Triplet loss
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0))
        return loss


def setup_wandb(project_name: str = "video-similarity-learning",
                experiment_name: str = None,
                config: Dict = None) -> None:
    """
    Setup Weights & Biases for experiment tracking.
    
    Args:
        project_name: Name of the W&B project
        experiment_name: Name of the experiment
        config: Configuration dictionary to log
    """
    if experiment_name is None:
        experiment_name = f"experiment_{wandb.util.generate_id()}"
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config
    )
    print(f"W&B initialized: {project_name}/{experiment_name}")


def create_optimizer(model: nn.Module, 
                    optimizer_type: str = 'adam',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'step',
                    step_size: int = 10,
                    gamma: float = 0.1) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('step', 'cosine')
        step_size: Step size for step scheduler
        gamma: Gamma for step scheduler
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=100)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                model_type: str = 'similarity') -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        model_type: Type of model ('similarity' or 'classification')
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        if model_type == 'similarity':
            frames1 = batch['frames1'].to(device)
            frames2 = batch['frames2'].to(device)
            labels = batch['similarity'].to(device)
            
            # Forward pass
            similarity = model(frames1, frames2)
            loss = criterion(similarity, labels)
            
            # Store predictions and targets
            predictions.extend(similarity.squeeze().detach().cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        else:  # classification
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(frames)
            loss = criterion(logits, labels)
            
            # Store predictions and targets
            pred_labels = torch.argmax(logits, dim=1)
            predictions.extend(pred_labels.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions) if model_type == 'classification' else None
    
    metrics = {
        'train_loss': avg_loss,
        'train_accuracy': accuracy
    }
    
    # Log to W&B
    wandb.log(metrics)
    
    return metrics


def validate_epoch(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  model_type: str = 'similarity') -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        model_type: Type of model ('similarity' or 'classification')
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if model_type == 'similarity':
                frames1 = batch['frames1'].to(device)
                frames2 = batch['frames2'].to(device)
                labels = batch['similarity'].to(device)
                
                # Forward pass
                similarity = model(frames1, frames2)
                loss = criterion(similarity, labels)
                
                # Store predictions and targets
                predictions.extend(similarity.squeeze().cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
            else:  # classification
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                logits = model(frames)
                loss = criterion(logits, labels)
                
                # Store predictions and targets
                pred_labels = torch.argmax(logits, dim=1)
                predictions.extend(pred_labels.cpu().numpy())
                targets.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions) if model_type == 'classification' else None
    
    metrics = {
        'val_loss': avg_loss,
        'val_accuracy': accuracy
    }
    
    # Additional metrics for similarity learning
    if model_type == 'similarity':
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Convert similarity scores to binary predictions
        pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(targets, pred_binary, average='binary')
        
        # Calculate AUC
        try:
            auc = roc_auc_score(targets, predictions)
        except:
            auc = 0.0
        
        metrics.update({
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_auc': auc
        })
    
    # Log to W&B
    wandb.log(metrics)
    
    return metrics


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filepath: str) -> Tuple[int, float]:
    """
    Load a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def train_model(model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                num_epochs: int,
                model_type: str = 'similarity',
                checkpoint_dir: str = 'checkpoints',
                save_freq: int = 5) -> Dict[str, List[float]]:
    """
    Complete training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs
        model_type: Type of model
        checkpoint_dir: Directory to save checkpoints
        save_freq: Frequency to save checkpoints
        
    Returns:
        Dictionary with training history
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, model_type)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, model_type)
        
        # Update history
        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        if train_metrics['train_accuracy'] is not None:
            history['train_accuracy'].append(train_metrics['train_accuracy'])
        if val_metrics['val_accuracy'] is not None:
            history['val_accuracy'].append(val_metrics['val_accuracy'])
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_metrics['val_loss'], checkpoint_path)
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch + 1, val_metrics['val_loss'], best_model_path)
    
    return history 