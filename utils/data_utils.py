"""
Data utilities for the Video Similarity Learning project.
This module provides dataset classes and data loading utilities.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import random
from .video_utils import load_video, create_frame_transforms, frames_to_tensor


class VideoDataset(Dataset):
    """
    Dataset class for video similarity learning.
    """
    
    def __init__(self, 
                 data_dir: str,
                 metadata_file: str,
                 max_frames: int = 30,
                 image_size: int = 224,
                 is_training: bool = True,
                 transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing video files
            metadata_file: Path to metadata CSV file
            max_frames: Maximum number of frames to extract per video
            image_size: Size to resize frames to
            is_training: Whether this is for training
            transform: Optional custom transforms
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.image_size = image_size
        self.is_training = is_training
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Create transforms
        if transform is None:
            self.transform = create_frame_transforms(image_size, is_training)
        else:
            self.transform = transform
        
        # Create video ID to path mapping
        self.video_paths = {}
        for _, row in self.metadata.iterrows():
            video_id = row['video_id']
            video_path = os.path.join(data_dir, f"{video_id}.mp4")
            if os.path.exists(video_path):
                self.video_paths[video_id] = video_path
        
        # Filter metadata to only include videos that exist
        self.metadata = self.metadata[self.metadata['video_id'].isin(self.video_paths.keys())]
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} videos from {data_dir}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_id = row['video_id']
        video_path = self.video_paths[video_id]
        
        # Load video frames
        frames = load_video(video_path, self.max_frames)
        
        # Convert to tensor
        frames_tensor = frames_to_tensor(frames, self.transform)
        
        # Get label (if available)
        label = row.get('label', 0)  # Default to 0 if no label
        
        return {
            'video_id': video_id,
            'frames': frames_tensor,
            'label': label,
            'video_path': video_path
        }


class VideoPairDataset(Dataset):
    """
    Dataset for training with video pairs (similarity learning).
    """
    
    def __init__(self, 
                 data_dir: str,
                 metadata_file: str,
                 max_frames: int = 30,
                 image_size: int = 224,
                 transform=None):
        """
        Initialize the pair dataset.
        
        Args:
            data_dir: Directory containing video files
            metadata_file: Path to metadata CSV file with similarity information
            max_frames: Maximum number of frames to extract per video
            image_size: Size to resize frames to
            transform: Optional custom transforms
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.image_size = image_size
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Create transforms
        if transform is None:
            self.transform = create_frame_transforms(image_size, is_training=True)
        else:
            self.transform = transform
        
        # Create video ID to path mapping
        self.video_paths = {}
        for _, row in self.metadata.iterrows():
            for col in ['video1_id', 'video2_id']:
                if col in row:
                    video_id = row[col]
                    video_path = os.path.join(data_dir, f"{video_id}.mp4")
                    if os.path.exists(video_path):
                        self.video_paths[video_id] = video_path
        
        # Filter metadata to only include videos that exist
        self.metadata = self.metadata[
            (self.metadata['video1_id'].isin(self.video_paths.keys())) &
            (self.metadata['video2_id'].isin(self.video_paths.keys()))
        ]
        self.metadata = self.metadata.reset_index(drop=True)
        
        print(f"Loaded {len(self.metadata)} video pairs from {data_dir}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video1_id = row['video1_id']
        video2_id = row['video2_id']
        similarity = row['similarity']  # 1 for similar, 0 for different
        
        # Load both videos
        video1_path = self.video_paths[video1_id]
        video2_path = self.video_paths[video2_id]
        
        frames1 = load_video(video1_path, self.max_frames)
        frames2 = load_video(video2_path, self.max_frames)
        
        # Convert to tensors
        frames1_tensor = frames_to_tensor(frames1, self.transform)
        frames2_tensor = frames_to_tensor(frames2, self.transform)
        
        return {
            'video1_id': video1_id,
            'video2_id': video2_id,
            'frames1': frames1_tensor,
            'frames2': frames2_tensor,
            'similarity': torch.tensor(similarity, dtype=torch.float32)
        }


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 8, 
                     shuffle: bool = True, 
                     num_workers: int = 4) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def create_sample_dataset(data_dir: str, num_samples: int = 100) -> pd.DataFrame:
    """
    Create a sample dataset for testing/development.
    
    Args:
        data_dir: Directory to create sample data in
        num_samples: Number of sample videos to create
        
    Returns:
        DataFrame with sample metadata
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample metadata
    sample_data = []
    for i in range(num_samples):
        video_id = f"sample_video_{i:04d}"
        label = random.randint(0, 9)  # Random label 0-9
        
        sample_data.append({
            'video_id': video_id,
            'label': label,
            'duration': random.uniform(5, 30),  # Random duration
            'source': 'sample'
        })
    
    metadata_df = pd.DataFrame(sample_data)
    metadata_path = os.path.join(data_dir, 'sample_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Created sample metadata with {num_samples} videos at {metadata_path}")
    return metadata_df


def create_similarity_pairs(metadata_file: str, 
                           output_file: str, 
                           num_pairs: int = 1000,
                           positive_ratio: float = 0.5) -> pd.DataFrame:
    """
    Create pairs of videos for similarity learning.
    
    Args:
        metadata_file: Path to video metadata file
        output_file: Path to save similarity pairs
        num_pairs: Number of pairs to create
        positive_ratio: Ratio of positive (similar) pairs
        
    Returns:
        DataFrame with similarity pairs
    """
    metadata = pd.read_csv(metadata_file)
    
    # Group videos by label
    label_groups = metadata.groupby('label')
    
    pairs_data = []
    num_positive = int(num_pairs * positive_ratio)
    num_negative = num_pairs - num_positive
    
    # Create positive pairs (same label)
    positive_pairs = 0
    for label, group in label_groups:
        if len(group) >= 2 and positive_pairs < num_positive:
            # Sample pairs from same label
            for _ in range(min(num_positive - positive_pairs, len(group) // 2)):
                pair = group.sample(n=2)
                pairs_data.append({
                    'video1_id': pair.iloc[0]['video_id'],
                    'video2_id': pair.iloc[1]['video_id'],
                    'similarity': 1,
                    'label1': label,
                    'label2': label
                })
                positive_pairs += 1
    
    # Create negative pairs (different labels)
    negative_pairs = 0
    labels = list(label_groups.groups.keys())
    
    while negative_pairs < num_negative:
        # Sample two different labels
        label1, label2 = random.sample(labels, 2)
        group1 = label_groups.get_group(label1)
        group2 = label_groups.get_group(label2)
        
        # Sample one video from each group
        video1 = group1.sample(n=1).iloc[0]
        video2 = group2.sample(n=1).iloc[0]
        
        pairs_data.append({
            'video1_id': video1['video_id'],
            'video2_id': video2['video_id'],
            'similarity': 0,
            'label1': label1,
            'label2': label2
        })
        negative_pairs += 1
    
    pairs_df = pd.DataFrame(pairs_data)
    pairs_df.to_csv(output_file, index=False)
    
    print(f"Created {len(pairs_df)} similarity pairs: {positive_pairs} positive, {negative_pairs} negative")
    return pairs_df 