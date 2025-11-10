import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image
import json
import os
from tqdm import tqdm

# ---
# 1. IMPORT YOUR MODEL
# ---
# Assumes your model class is in 'tracker_model.py'
try:
    from model import ZaloTrackerNet
except ImportError:
    print("FATAL: Could not find 'tracker_model.py'.")
    print("Please paste your ZaloTrackerNet class into a file named 'tracker_model.py' in this directory.")
    exit()

# ---
# 2. HELPER FUNCTION FOR BOX CONVERSION
# ---

def box_cxcywh_to_xyxy(x):
    """
    Converts (cx, cy, w, h) normalized boxes to (x1, y1, x2, y2)
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# ---
# 3. INFERENCE CONFIGURATION
# ---

# --- !! SET YOUR PATHS HERE !! ---
CHECKPOINT_PATH = "/kaggle/input/test-tracking-model/pytorch/default/1/final_model_epoch400.pth" # Your trained model
QUERY_IMAGE_PATH = "/kaggle/input/object-image-test/img_1.jpg"
VIDEO_PATH = "/kaggle/input/test-video/drone_video.mp4"
JSON_OUTPUT_PATH = "/output_submission.json"
# ---------------------------------

# --- Hyperparameters ---
# This is the training clip size. DO NOT CHANGE.
CLIP_LENGTH = 32
# You can tune this threshold to trade off precision/recall
SCORE_THRESHOLD = 0.5 

@torch.no_grad()
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    print("Loading model...")
    model = ZaloTrackerNet().to(device)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    except FileNotFoundError:
        print(f"FATAL: Checkpoint file not found at {CHECKPOINT_PATH}")
        return
    model.eval()

    # 2. Define Transforms (MUST match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Load and Process Query Image
    print(f"Loading query image from {QUERY_IMAGE_PATH}...")
    try:
        query_img = Image.open(QUERY_IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"FATAL: Query image not found at {QUERY_IMAGE_PATH}")
        return

    query_tensor = transform(query_img)
    
    # Add the [1, C, H, W] dimension (from dataset __getitem__)
    query_tensor = query_tensor.unsqueeze(0) 
    
    # Add the Batch dimension [B, 1, C, H, W] (from collate_fn)
    # This simulates a batch size of 1 for the model's forward pass.
    query_tensor = query_tensor.unsqueeze(0).to(device)
    
    print(f"Query tensor shape: {query_tensor.shape}") # Should be [1, 1, 3, 224, 224]

    # 4. Open Video and Get Properties
    print(f"Opening video {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"FATAL: Could not open video file {VIDEO_PATH}")
        return
        
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {video_width}W x {video_height}H, {total_frames} total frames.")

    # 5. Process Video in Clips
    clip_frames = []
    frame_indices = []
    all_detected_bboxes = [] # This will store the final JSON output bboxes

    pbar = tqdm(total=total_frames, desc="Processing video")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        # Convert frame from BGR (cv2) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Transform and store
        clip_frames.append(transform(pil_img))
        frame_indices.append(frame_idx)
        pbar.update(1)

        # When clip is full, process it
        if len(clip_frames) == CLIP_LENGTH:
            # --- Process this clip ---
            # Stack frames into a [T, C, H, W] tensor, add Batch dim [B, T, C, H, W]
            video_tensor = torch.stack(clip_frames).unsqueeze(0).to(device)
            
            # Model forward pass
            outputs = model(query_tensor, video_tensor)
            
            # Post-process the outputs
            # Squeeze batch dim [1, T, 49, 1] -> [T, 49, 1]
            pred_logits = outputs['pred_logits'].squeeze(0)
            pred_boxes = outputs['pred_boxes'].squeeze(0)

            # Get scores [T, 49] by squeezing logit dim and applying sigmoid
            scores = pred_logits.squeeze(-1).sigmoid() 
            
            # Find best score for each frame in the clip
            best_scores_per_frame, best_indices_per_frame = scores.max(dim=1)
            
            for i in range(len(frame_indices)):
                score = best_scores_per_frame[i].item()
                
                # Only add if score is above threshold
                if score >= SCORE_THRESHOLD:
                    box_idx = best_indices_per_frame[i]
                    norm_box = pred_boxes[i, box_idx, :] # [cx, cy, w, h]
                    
                    # Convert normalized [cx,cy,w,h] to absolute [x1,y1,x2,y2]
                    abs_box = box_cxcywh_to_xyxy(norm_box)
                    abs_box[0] *= video_width
                    abs_box[1] *= video_height
                    abs_box[2] *= video_width
                    abs_box[3] *= video_height
                    
                    # Clamp to image boundaries
                    abs_box = abs_box.clamp(min=0)
                    
                    # Format for JSON
                    bbox_dict = {
                        "frame": int(frame_indices[i]),
                        "x1": int(abs_box[0].item()),
                        "y1": int(abs_box[1].item()),
                        "x2": int(abs_box[2].item()),
                        "y2": int(abs_box[3].item())
                    }
                    all_detected_bboxes.append(bbox_dict)

            # Clear lists for next clip
            clip_frames = []
            frame_indices = []
            # --- End of clip processing ---

    # 6. Process the last partial clip
    if clip_frames:
        print(f"\nProcessing final partial clip of {len(clip_frames)} frames...")
        # Stack frames and create batch dim
        video_tensor = torch.stack(clip_frames).unsqueeze(0).to(device)
        
        # Model forward pass
        outputs = model(query_tensor, video_tensor)
        
        # Post-process
        pred_logits = outputs['pred_logits'].squeeze(0)
        pred_boxes = outputs['pred_boxes'].squeeze(0)
        scores = pred_logits.squeeze(-1).sigmoid() 
        best_scores_per_frame, best_indices_per_frame = scores.max(dim=1)
        
        for i in range(len(frame_indices)):
            score = best_scores_per_frame[i].item()
            if score >= SCORE_THRESHOLD:
                box_idx = best_indices_per_frame[i]
                norm_box = pred_boxes[i, box_idx, :]
                abs_box = box_cxcywh_to_xyxy(norm_box)
                abs_box[0] *= video_width
                abs_box[1] *= video_height
                abs_box[2] *= video_width
                abs_box[3] *= video_height
                abs_box = abs_box.clamp(min=0)
                
                bbox_dict = {
                    "frame": int(frame_indices[i]),
                    "x1": int(abs_box[0].item()),
                    "y1": int(abs_box[1].item()),
                    "x2": int(abs_box[2].item()),
                    "y2": int(abs_box[3].item())
                }
                all_detected_bboxes.append(bbox_dict)
    
    pbar.close()
    cap.release()
    print("Video processing complete.")

    # 7. Format and Save JSON Output
    video_id = os.path.basename(VIDEO_PATH).split('.')[0]
    
    # This structure matches your example exactly
    output_data = [
      {
        "video_id": video_id,
        "annotations": [
          {
            "bboxes": all_detected_bboxes
          }
        ]
      }
    ]

    print(f"Saving {len(all_detected_bboxes)} bounding boxes to {JSON_OUTPUT_PATH}...")
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Inference finished. Output saved to {JSON_OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference()