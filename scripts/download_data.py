#!/usr/bin/env python3
"""
Data download script for the Video Similarity Learning project.
This script creates sample data for educational purposes.
"""

import os
import sys
import requests
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import create_sample_dataset, create_similarity_pairs


def download_sample_videos(data_dir: str, num_videos: int = 50) -> None:
    """
    Download sample videos for educational purposes.
    For this educational project, we'll create synthetic video data.
    
    Args:
        data_dir: Directory to save videos
        num_videos: Number of videos to create
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Creating {num_videos} sample videos...")
    
    # Create synthetic videos with different patterns
    for i in tqdm(range(num_videos), desc="Creating videos"):
        video_id = f"sample_video_{i:04d}"
        video_path = os.path.join(data_dir, f"{video_id}.mp4")
        
        # Create a simple synthetic video
        create_synthetic_video(video_path, duration=5, fps=10, pattern=i % 5)
    
    print(f"Created {num_videos} sample videos in {data_dir}")


def create_synthetic_video(video_path: str, duration: int = 5, fps: int = 10, pattern: int = 0) -> None:
    """
    Create a synthetic video for educational purposes.
    
    Args:
        video_path: Path to save the video
        duration: Duration in seconds
        fps: Frames per second
        pattern: Pattern type (0-4 for different visual patterns)
    """
    # Video parameters
    width, height = 320, 240
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Create frame based on pattern
        frame = create_pattern_frame(width, height, frame_idx, pattern, total_frames)
        
        # Write frame
        out.write(frame)
    
    out.release()


def create_pattern_frame(width: int, height: int, frame_idx: int, pattern: int, total_frames: int) -> np.ndarray:
    """
    Create a frame with a specific pattern.
    
    Args:
        width: Frame width
        height: Frame height
        frame_idx: Current frame index
        pattern: Pattern type
        total_frames: Total number of frames
        
    Returns:
        Frame as numpy array
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Normalize frame index
    t = frame_idx / total_frames
    
    if pattern == 0:
        # Moving circle
        center_x = int(width * (0.2 + 0.6 * t))
        center_y = int(height * 0.5)
        radius = 30
        cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), -1)
        
    elif pattern == 1:
        # Color changing rectangle
        color = (
            int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t))),
            int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 2 * np.pi / 3))),
            int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * t + 4 * np.pi / 3)))
        )
        x1, y1 = int(width * 0.2), int(height * 0.2)
        x2, y2 = int(width * 0.8), int(height * 0.8)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
    elif pattern == 2:
        # Rotating triangle
        center_x, center_y = width // 2, height // 2
        angle = 2 * np.pi * t
        size = 40
        
        # Calculate triangle vertices
        points = []
        for i in range(3):
            point_angle = angle + i * 2 * np.pi / 3
            x = int(center_x + size * np.cos(point_angle))
            y = int(center_y + size * np.sin(point_angle))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(frame, [points], (0, 255, 0))
        
    elif pattern == 3:
        # Expanding squares
        max_size = min(width, height) // 4
        size = int(max_size * (0.2 + 0.8 * t))
        x1 = (width - size) // 2
        y1 = (height - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
    else:  # pattern == 4
        # Wave pattern
        for x in range(0, width, 10):
            y = int(height * 0.5 + 50 * np.sin(2 * np.pi * (x / width + t)))
            cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)
    
    return frame


def create_metadata_files(data_dir: str) -> None:
    """
    Create metadata files for the sample dataset.
    
    Args:
        data_dir: Directory containing the videos
    """
    print("Creating metadata files...")
    
    # Create main metadata
    metadata = create_sample_dataset(data_dir, num_samples=50)
    
    # Create similarity pairs
    metadata_file = os.path.join(data_dir, 'sample_metadata.csv')
    pairs_file = os.path.join(data_dir, 'similarity_pairs.csv')
    
    create_similarity_pairs(
        metadata_file=metadata_file,
        output_file=pairs_file,
        num_pairs=200,
        positive_ratio=0.5
    )
    
    print(f"Created metadata files:")
    print(f"  - {metadata_file}")
    print(f"  - {pairs_file}")


def download_pretrained_models(models_dir: str) -> None:
    """
    Download pre-trained models (for educational purposes, we'll create placeholder files).
    
    Args:
        models_dir: Directory to save models
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("Setting up pre-trained model placeholders...")
    
    # Create placeholder files for pre-trained models
    model_files = [
        'resnet50_pretrained.pth',
        'resnet101_pretrained.pth',
        'resnet18_pretrained.pth'
    ]
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            # Create a placeholder file
            with open(model_path, 'w') as f:
                f.write("# This is a placeholder for educational purposes.\n")
                f.write("# In a real project, this would contain pre-trained weights.\n")
            print(f"Created placeholder: {model_path}")
    
    print("Note: For real training, you would need to download actual pre-trained models.")


def main():
    """Main function to download and setup all data."""
    print("=== Video Similarity Learning - Data Setup ===")
    
    # Create data directory structure
    data_dir = os.path.join(project_root, "data")
    videos_dir = os.path.join(data_dir, "videos")
    models_dir = os.path.join(data_dir, "models")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Setting up data in: {data_dir}")
    
    # Download/create sample videos
    download_sample_videos(videos_dir, num_videos=50)
    
    # Create metadata files
    create_metadata_files(videos_dir)
    
    # Setup pre-trained models
    download_pretrained_models(models_dir)
    
    print("\n=== Data Setup Complete ===")
    print(f"Data directory: {data_dir}")
    print(f"Videos: {videos_dir}")
    print(f"Models: {models_dir}")
    print("\nYou can now start with the Jupyter notebooks!")


if __name__ == "__main__":
    main() 