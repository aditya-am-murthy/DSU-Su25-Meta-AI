"""
Video processing utilities for the Video Similarity Learning project.
This module provides functions for loading videos and extracting frames.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch
from torchvision import transforms


def load_video(video_path: str, max_frames: int = 30) -> np.ndarray:
    """
    Load a video and extract frames.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        
    Returns:
        Array of frames with shape (num_frames, height, width, channels)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to extract (uniformly distributed)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                break
                
    finally:
        cap.release()
    
    return np.array(frames)


def extract_frames(video_path: str, output_dir: str, max_frames: int = 30) -> List[str]:
    """
    Extract frames from a video and save them as images.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract
        
    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frames = load_video(video_path, max_frames)
    frame_paths = []
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video information
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def create_frame_transforms(image_size: int = 224, is_training: bool = True) -> transforms.Compose:
    """
    Create transforms for video frames.
    
    Args:
        image_size: Size to resize images to
        is_training: Whether transforms are for training (includes augmentation)
        
    Returns:
        torchvision transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def frames_to_tensor(frames: np.ndarray, transform=None) -> torch.Tensor:
    """
    Convert frames array to tensor with optional transforms.
    
    Args:
        frames: Array of frames with shape (num_frames, height, width, channels)
        transform: Optional transform to apply to each frame
        
    Returns:
        Tensor with shape (num_frames, channels, height, width)
    """
    if transform is None:
        transform = create_frame_transforms(is_training=False)
    
    frame_tensors = []
    for frame in frames:
        frame_tensor = transform(frame)
        frame_tensors.append(frame_tensor)
    
    return torch.stack(frame_tensors)


def visualize_frames(frames: np.ndarray, num_frames: int = 8) -> None:
    """
    Visualize frames from a video (for debugging/exploration).
    
    Args:
        frames: Array of frames
        num_frames: Number of frames to display
    """
    import matplotlib.pyplot as plt
    
    if len(frames) < num_frames:
        num_frames = len(frames)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_frames):
        frame_idx = int(i * len(frames) / num_frames)
        axes[i].imshow(frames[frame_idx])
        axes[i].set_title(f'Frame {frame_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show() 