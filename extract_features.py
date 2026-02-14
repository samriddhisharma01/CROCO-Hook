import cv2
import torch
import numpy as np
import os
import glob
from model_utils import get_dino_model, get_transform

# --- CONFIGURATION ---
INPUT_FOLDER = "data"    # Put your mp4s here
OUTPUT_FOLDER = "output" # Where .npz files will be saved
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure folders exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load shared tools from model_utils
print(f"Loading DINOv2 on {device}...")
dino = get_dino_model(device)
transform = get_transform()

# Find all mp4 files
video_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))

if not video_paths:
    print(f"No .mp4 files found in '{INPUT_FOLDER}'.")

for video_path in video_paths:
    video_name = os.path.basename(video_path)
    print(f"Processing: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    all_timestamps = []
    all_dino = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get Timestamp (seconds)
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        all_timestamps.append(t)

        # Get DINO Embeddings 
        img = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = dino(img).cpu().numpy()[0]
        all_dino.append(emb)

    cap.release()

    # Save output
    save_name = video_name.replace(".mp4", "_features.npz")
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    
    np.savez_compressed(
        save_path,
        timestamps=np.array(all_timestamps),
        dino=np.stack(all_dino)
    )
    print(f"Successfully saved features to: {save_path}")

print("Extraction Complete!")