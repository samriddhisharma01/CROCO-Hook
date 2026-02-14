import torch
import torch.nn as nn
import numpy as np
import json
import os
import glob
from torch.utils.data import Dataset, DataLoader
from model_utils import BiGRUResNet

# --- CONFIGURATION ---
FEATURE_DIR = "output"  # Directory containing .npz files
LABEL_DIR = "data"      # Directory containing .json files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_video_configs(feature_dir, label_dir):
    """
    Dynamically pairs feature files with their corresponding boundary JSONs.
    Assumes naming convention: 'video1_features.npz' matches 'video1_labels.json'
    or your current 'boundariesX.json' pattern.
    """
    configs = []
    feature_files = glob.glob(os.path.join(feature_dir, "*.npz"))
    
    for npz_path in feature_files:
        base_name = os.path.basename(npz_path).split('_')[0]
        
        # Look for matching json: tries base_name_labels.json or boundariesX.json
        json_pattern = os.path.join(label_dir, f"{base_name}*.json")
        json_matches = glob.glob(json_pattern)
        
        if json_matches:
            configs.append({
                "npz": npz_path,
                "json": json_matches[0]
            })
    return configs

class CrochetBoundaryDataset(Dataset):
    def __init__(self, video_configs, window_size=64, stride=16, sigma=0.25):
        self.samples = []
        
        for config in video_configs:
            # 1. Load data
            data = np.load(config['npz'])
            with open(config['json'], 'r') as f:
                boundary_ts = json.load(f)["boundaries_seconds"]
            
            dino = data['dino']          # Shape: (T, 384)
            timestamps = data['timestamps']
            
            soft_labels = np.zeros(len(timestamps))

            for ts in boundary_ts:
                time_diff = np.abs(timestamps - ts)
                gaussian = np.exp(-(time_diff ** 2) / (2 * sigma ** 2))
                soft_labels = np.maximum(soft_labels, gaussian)

            # 3. Create Windows from ALL videos
            # 3. Create regular sliding windows
            for i in range(0, len(dino) - window_size, stride):
                self.samples.append({
                    'x': torch.FloatTensor(dino[i : i + window_size]),
                    'y': torch.FloatTensor(soft_labels[i : i + window_size])
                })
            
            # 4. Left-biased final sweep (extra tail coverage, no padding)
            T = len(dino)
            tail_starts = [
                T - window_size - stride,
                T - window_size - 2 * stride
            ]
            
            for i in tail_starts:
                if i >= 0:
                    self.samples.append({
                        'x': torch.FloatTensor(dino[i : i + window_size]),
                        'y': torch.FloatTensor(soft_labels[i : i + window_size])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]['x'], self.samples[idx]['y']

# --- EXECUTION ---

# Get paired files
video_configs = get_video_configs(FEATURE_DIR, LABEL_DIR)
if not video_configs:
    raise FileNotFoundError("No paired feature and label files found.")

dataset = CrochetBoundaryDataset(video_configs)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20

model = BiGRUResNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCELoss()

print(f"Training on {len(video_configs)} videos...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x = x.to(DEVICE)   # (B, T, 384)
        y = y.to(DEVICE)   # (B, T)

        optimizer.zero_grad()
        preds = model(x)   # (B, T)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {epoch_loss / len(train_loader):.4f}")

