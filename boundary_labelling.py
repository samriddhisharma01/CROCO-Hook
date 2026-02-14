import cv2
import time
import json
import os
import glob

# --- CONFIGURATION ---
# Points to the same data folder used for extraction
INPUT_FOLDER = "data" 

# Search for the first video in the folder to label
video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4"))

if not video_files:
    print("No videos found to label. Check your 'data' folder.")
    exit()

# Change index [0] to label different videos
video_to_label = video_files[0] 
print(f"Labelling: {video_to_label}")
print("Controls: 'b' to mark boundary, 'q' to save and quit.")

cap = cv2.VideoCapture(video_to_label)
prev_time = None
boundaries = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    # Sync playback speed
    if prev_time is not None:
        dt = t - prev_time
        if dt > 0:
            time.sleep(dt)
    prev_time = t

    cv2.imshow("Crochet Labeling Tool", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('b'):
        print(f"Stitch Boundary recorded at {t:.3f}s")
        boundaries.append(round(t, 3))

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save labels with a name corresponding to the video
json_name = os.path.basename(video_to_label).replace(".mp4", "_labels.json")
with open(json_name, "w") as f:
    json.dump({"boundaries_seconds": boundaries}, f, indent=2)

print(f"Saved labels to {json_name}")