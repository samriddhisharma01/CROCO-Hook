import cv2
import torch
import numpy as np
import os
import glob
from torchvision import transforms

# --- CONFIGURATION ---
INPUT_FOLDER = "data"    # Folder containing your .mp4 files
OUTPUT_FOLDER = "output" # Where .npz files will be saved
MODEL_TYPE = "dinov2_vits14"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load DINOv2
print(f"Loading {MODEL_TYPE} on {device}...")
dino = torch.hub.load("facebookresearch/dinov2", MODEL_TYPE)
dino.eval()
dino.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Find all mp4 files in the data folder
video_paths = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))

if not video_paths:
    print(f"No .mp4 files found in '{INPUT_FOLDER}'. Please add some videos!")

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

    # Save output to the output folder
    save_name = video_name.replace(".mp4", "_features.npz")
    save_path = os.path.join(OUTPUT_FOLDER, save_name)
    
    np.savez_compressed(
        save_path,
        timestamps=np.array(all_timestamps),
        dino=np.stack(all_dino)
    )
    print(f"Successfully saved features to: {save_path}")

print("Done!")
