"""
06_Final_Project.py - Final Deliverable: Custom Video Similarity Model

This script is your final project. You will:
- Design and implement your own model architecture
- Load and fine-tune pretrained weights (if desired)
- Train and evaluate your model on the video similarity dataset
- Experiment with different approaches and report your results

Instructions:
- Fill in the TODO sections with your own code and design choices
- Use this script as a template for industry-style ML experimentation
- Submit your completed script and a short report on your findings
"""

import sys, os
from pathlib import Path
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# === Project Setup ===
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import VideoDataset
from utils.model_utils import VideoSiameseNetwork, VideoTripletNetwork
from utils.training_utils import ContrastiveLoss, TripletLoss

# === Argument Parsing ===
def parse_args():
    parser = argparse.ArgumentParser(description="Custom Video Similarity Model Final Project")
    parser.add_argument('--config', type=str, default=str(project_root / 'configs' / 'default_config.yaml'), help='Path to config YAML')
    parser.add_argument('--wandb_project', type=str, default='video-similarity-final', help='wandb project name')
    parser.add_argument('--student_name', type=str, default='student', help='Your name for experiment tracking')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model weights (optional)')
    parser.add_argument('--custom_model', type=str, default=None, help='Path to custom model definition (optional)')
    return parser.parse_args()

# === Load Config ===
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# === Data Preparation ===
def prepare_data(config):
    data_dir = project_root / 'data' / 'videos'
    metadata_file = data_dir / 'sample_metadata.csv'
    pairs_file = data_dir / 'similarity_pairs.csv'
    metadata = pd.read_csv(metadata_file)
    pairs = pd.read_csv(pairs_file)
    from sklearn.model_selection import train_test_split
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.2, random_state=42, stratify=pairs['similarity']
    )
    train_dataset = VideoDataset(
        pairs=train_pairs, video_dir=data_dir,
        max_frames=config['data']['max_frames'],
        image_size=config['data']['image_size'], is_training=True)
    val_dataset = VideoDataset(
        pairs=val_pairs, video_dir=data_dir,
        max_frames=config['data']['max_frames'],
        image_size=config['data']['image_size'], is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2)
    return train_loader, val_loader

# === Model Definition ===
def get_model(config, custom_model_path=None):
    # TODO: Students: Replace or extend this function to define your own architecture
    if custom_model_path:
        # Example: load a custom model definition from a .py file
        import importlib.util
        spec = importlib.util.spec_from_file_location("CustomModel", custom_model_path)
        custom_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_mod)
        return custom_mod.CustomModel(config)
    if config['model']['architecture'] == 'siamese':
        return VideoSiameseNetwork(
            feature_dim=config['model']['feature_dim'],
            embedding_dim=config['model']['embedding_dim'])
    elif config['model']['architecture'] == 'triplet':
        return VideoTripletNetwork(
            feature_dim=config['model']['feature_dim'],
            embedding_dim=config['model']['embedding_dim'])
    else:
        raise ValueError('Unknown architecture')

# === Training and Evaluation ===
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc='Train'):
        video1, video2, labels = batch
        video1, video2, labels = video1.to(device), video2.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(video1, video2)
        loss = loss_fn(preds, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            video1, video2, labels = batch
            video1, video2, labels = video1.to(device), video2.to(device), labels.to(device)
            preds = model(video1, video2)
            loss = loss_fn(preds, labels.float())
            total_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return total_loss / len(loader), np.concatenate(all_preds), np.concatenate(all_labels)

# === Main Script ===
def main():
    args = parse_args()
    config = load_config(args.config)
    train_loader, val_loader = prepare_data(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === wandb Setup ===
    wandb.init(project=args.wandb_project, config=config, tags=[args.student_name, config['model']['architecture']])
    wandb.config.student_name = args.student_name

    # === Model ===
    model = get_model(config, custom_model_path=args.custom_model)
    model = model.to(device)
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))

    # === Loss and Optimizer ===
    if config['model']['architecture'] == 'siamese':
        loss_fn = ContrastiveLoss(margin=config['training']['margin'])
    else:
        loss_fn = TripletLoss(margin=config['training']['margin'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # === Training Loop ===
    n_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_labels = validate_one_epoch(model, val_loader, loss_fn, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f'Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        # TODO: Add your own metrics (accuracy, ROC AUC, etc.) and log to wandb
        # TODO: Add early stopping, checkpointing, or other enhancements
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{args.student_name}.pt")
            print("Best model saved!")

    # === Final Evaluation ===
    # TODO: Add your own evaluation code (metrics, confusion matrix, etc.)
    print("Training complete. Evaluate your model and report your findings!")

if __name__ == "__main__":
    main() 