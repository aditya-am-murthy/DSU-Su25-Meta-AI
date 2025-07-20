"""
Model utilities for the Video Similarity Learning project.
This module provides functions for creating and managing model architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import numpy as np


class VideoFeatureExtractor(nn.Module):
    """
    Feature extractor for video frames using pre-trained models.
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 feature_dim: int = 2048,
                 num_frames: int = 30,
                 pooling_method: str = 'mean'):
        """
        Initialize the feature extractor.
        
        Args:
            backbone: Pre-trained model backbone ('resnet50', 'resnet101', 'vit', etc.)
            feature_dim: Output feature dimension
            num_frames: Number of frames per video
            pooling_method: Method to pool frame features ('mean', 'max', 'attention')
        """
        super().__init__()
        
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.pooling_method = pooling_method
        
        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone_model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.backbone_model = nn.Sequential(*list(self.backbone_model.children())[:-1])
            backbone_output_dim = 2048
        elif backbone == 'resnet101':
            self.backbone_model = models.resnet101(pretrained=True)
            self.backbone_model = nn.Sequential(*list(self.backbone_model.children())[:-1])
            backbone_output_dim = 2048
        elif backbone == 'resnet18':
            self.backbone_model = models.resnet18(pretrained=True)
            self.backbone_model = nn.Sequential(*list(self.backbone_model.children())[:-1])
            backbone_output_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projection layer
        self.feature_projection = nn.Linear(backbone_output_dim, feature_dim)
        
        # Attention mechanism for temporal pooling
        if pooling_method == 'attention':
            self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(feature_dim)
        
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            frames: Input frames tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Video features tensor of shape (batch_size, feature_dim)
        """
        batch_size, num_frames = frames.shape[:2]
        
        # Reshape frames for batch processing
        frames_flat = frames.view(-1, *frames.shape[2:])  # (batch_size * num_frames, channels, height, width)
        
        # Extract features from each frame
        frame_features = self.backbone_model(frames_flat)  # (batch_size * num_frames, backbone_output_dim, 1, 1)
        frame_features = frame_features.squeeze(-1).squeeze(-1)  # (batch_size * num_frames, backbone_output_dim)
        
        # Project to desired feature dimension
        frame_features = self.feature_projection(frame_features)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal pooling
        if self.pooling_method == 'mean':
            video_features = torch.mean(frame_features, dim=1)  # (batch_size, feature_dim)
        elif self.pooling_method == 'max':
            video_features = torch.max(frame_features, dim=1)[0]  # (batch_size, feature_dim)
        elif self.pooling_method == 'attention':
            # Self-attention for temporal modeling
            attended_features, _ = self.attention(frame_features, frame_features, frame_features)
            video_features = torch.mean(attended_features, dim=1)  # (batch_size, feature_dim)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")
        
        # Apply batch normalization
        video_features = self.batch_norm(video_features)
        
        return video_features


class SimilarityModel(nn.Module):
    """
    Model for video similarity learning.
    """
    
    def __init__(self, 
                 feature_extractor: VideoFeatureExtractor,
                 similarity_method: str = 'cosine'):
        """
        Initialize the similarity model.
        
        Args:
            feature_extractor: Feature extractor for videos
            similarity_method: Method to compute similarity ('cosine', 'euclidean', 'learned')
        """
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.similarity_method = similarity_method
        
        if similarity_method == 'learned':
            # Learnable similarity function
            self.similarity_net = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
    
    def forward(self, frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between two videos.
        
        Args:
            frames1: Frames from first video (batch_size, num_frames, channels, height, width)
            frames2: Frames from second video (batch_size, num_frames, channels, height, width)
            
        Returns:
            Similarity scores (batch_size, 1)
        """
        # Extract features
        features1 = self.feature_extractor(frames1)  # (batch_size, feature_dim)
        features2 = self.feature_extractor(frames2)  # (batch_size, feature_dim)
        
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute similarity
        if self.similarity_method == 'cosine':
            similarity = torch.sum(features1 * features2, dim=1, keepdim=True)
        elif self.similarity_method == 'euclidean':
            distance = torch.norm(features1 - features2, p=2, dim=1, keepdim=True)
            similarity = torch.exp(-distance)  # Convert distance to similarity
        elif self.similarity_method == 'learned':
            # Concatenate features and pass through learned network
            combined_features = torch.cat([features1, features2], dim=1)
            similarity = self.similarity_net(combined_features)
        else:
            raise ValueError(f"Unsupported similarity method: {self.similarity_method}")
        
        return similarity


class ClassificationModel(nn.Module):
    """
    Model for video classification.
    """
    
    def __init__(self, 
                 feature_extractor: VideoFeatureExtractor,
                 num_classes: int):
        """
        Initialize the classification model.
        
        Args:
            feature_extractor: Feature extractor for videos
            num_classes: Number of classes
        """
        super().__init__()
        
        self.feature_extractor = feature_extractor
        self.classifier = nn.Sequential(
            nn.Linear(feature_extractor.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Classify video.
        
        Args:
            frames: Video frames (batch_size, num_frames, channels, height, width)
            
        Returns:
            Classification logits (batch_size, num_classes)
        """
        features = self.feature_extractor(frames)
        logits = self.classifier(features)
        return logits


def create_model(model_type: str = 'similarity',
                backbone: str = 'resnet50',
                feature_dim: int = 2048,
                num_frames: int = 30,
                pooling_method: str = 'mean',
                similarity_method: str = 'cosine',
                num_classes: int = 10) -> nn.Module:
    """
    Create a model based on the specified parameters.
    
    Args:
        model_type: Type of model ('similarity' or 'classification')
        backbone: Backbone architecture
        feature_dim: Feature dimension
        num_frames: Number of frames per video
        pooling_method: Temporal pooling method
        similarity_method: Similarity computation method
        num_classes: Number of classes (for classification)
        
    Returns:
        PyTorch model
    """
    feature_extractor = VideoFeatureExtractor(
        backbone=backbone,
        feature_dim=feature_dim,
        num_frames=num_frames,
        pooling_method=pooling_method
    )
    
    if model_type == 'similarity':
        model = SimilarityModel(feature_extractor, similarity_method)
    elif model_type == 'classification':
        model = ClassificationModel(feature_extractor, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (excluding batch dimension)
        
    Returns:
        Dictionary with model summary
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    # Test forward pass
    try:
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape)
            output = model(dummy_input)
            output_shape = output.shape
    except Exception as e:
        output_shape = f"Error: {str(e)}"
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'model_type': type(model).__name__
    }


def save_model(model: nn.Module, 
               filepath: str, 
               optimizer=None, 
               epoch: int = 0, 
               loss: float = 0.0) -> None:
    """
    Save a model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: Path to save the checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch
        loss: Current loss value
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")


def load_model(model: nn.Module, 
               filepath: str, 
               optimizer=None) -> Tuple[nn.Module, int, float]:
    """
    Load a model checkpoint.
    
    Args:
        model: PyTorch model
        filepath: Path to the checkpoint
        optimizer: Optimizer (optional)
        
    Returns:
        Tuple of (model, epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return model, epoch, loss 