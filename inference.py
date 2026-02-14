import cv2
import torch
import numpy as np
import os
from model_utils import BiGRUResNet, get_dino_model, get_transform

# --- CONFIGURATION ---
VIDEO_PATH = "data/your_video.mp4" 
MODEL_PATH = "models/windowbigru_boundary_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SETUP ---
dino_model = get_dino_model(DEVICE)
transform = get_transform()

model = BiGRUResNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle different checkpoint saving styles
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

# --- STEP 1: EXTRACT FEATURES ---
print(f"Extracting features from {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)
all_timestamps, all_dino = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    all_timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    img = transform(frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        all_dino.append(dino_model(img).cpu().numpy()[0])
cap.release()

dino_features, timestamps = np.stack(all_dino), np.array(all_timestamps)

# --- STEP 2: INFERENCE (STITCH COUNTING LOGIC) ---
stitch_count = 0
is_above_threshold = False
DETECTION_THRESHOLD, RELEASE_THRESHOLD = 0.1, 0.05
window_size, commit_start, commit_end = 64, 16, 32

T = len(dino_features)
rt_scores = np.full(T, np.nan)
buffer, commit_ptr = [], 0

print("Running detection...")
with torch.no_grad():
    for t in range(T):
        buffer.append(dino_features[t])
        if len(buffer) > window_size: buffer.pop(0)

        if len(buffer) == window_size:
            x = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(DEVICE)
            y = model(x).squeeze(0).cpu().numpy()

            window_start_at = t - window_size + 1
            global_start, global_end = window_start_at + commit_start, window_start_at + commit_end
            
            write_start = max(commit_ptr, global_start)
            if write_start < global_end:
                y_start = commit_start + (write_start - global_start)
                rt_scores[write_start:global_end] = y[y_start:commit_end]
                
                for idx in range(write_start, global_end):
                    current_val = rt_scores[idx]
                    if current_val > DETECTION_THRESHOLD and not is_above_threshold:
                        is_above_threshold = True
                    elif current_val < RELEASE_THRESHOLD and is_above_threshold:
                        stitch_count += 1
                        is_above_threshold = False
                        print(f"Stitch {stitch_count} detected. (Time: {timestamps[idx]:.2f}s)")
                commit_ptr = global_end

    # FLUSHING THE LAST WINDOW
    if len(buffer) > 0:
        x = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(DEVICE)
        y = model(x).squeeze(0).cpu().numpy()
        num_left = T - commit_ptr
        if num_left > 0:
            rt_scores[commit_ptr:] = y[-num_left:]
            for idx in range(commit_ptr, T):
                if rt_scores[idx] > DETECTION_THRESHOLD and not is_above_threshold:
                    is_above_threshold = True
                elif rt_scores[idx] < RELEASE_THRESHOLD and is_above_threshold:
                    stitch_count += 1
                    is_above_threshold = False
                    print(f"Stitch {stitch_count} detected. (Time: {timestamps[idx]:.2f}s)")

if is_above_threshold:
    stitch_count += 1
    print(f"Stitch {stitch_count} detected. (Time: {timestamps[-1]:.2f}s)")

print(f"\nFinal Total Count: {stitch_count}")