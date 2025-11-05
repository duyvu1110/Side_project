import os
import json
import torch
import random
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from box_utils import box_xyxy_to_cxcywh
# ---
from helper_function import load_video_frames_at_indices
# 1. ZALO AI DATASET CLASS (CORRECTED)
# ---

class ZaloAIDataset(Dataset):
    """
    Dataset for the Zalo AI Challenge data structure.
    CORRECTED version based on the new annotation format.
    """
    def __init__(self, root_dir, phase='train', num_frames=32):
        super(ZaloAIDataset, self).__init__()
        
        self.root_dir = root_dir
        self.phase = phase
        self.num_frames = num_frames
        
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.annos_path = os.path.join(self.root_dir, 'annotations', 'annotations.json')
        
        # 1. Get a list of all sample folders
        self.sample_names = os.listdir(self.samples_dir)
        self.sample_names = [s for s in self.sample_names if not s.startswith('.')]
        
        # 2. **NEW: Load and process the annotations**
        print(f"Loading annotations from {self.annos_path}...")
        try:
            with open(self.annos_path) as f:
                all_annos_list = json.load(f)
        except FileNotFoundError:
            print(f"FATAL ERROR: Annotation file not found at {self.annos_path}")
            raise
            
        # 3. **NEW: Convert annotations list to a fast-lookup dictionary**
        # This is the most important change.
        # We create a nested dict: {video_id: {frame_idx: [list of boxes]}}
        self.annos_dict = {}
        for item in all_annos_list:
            video_id = item['video_id']
            # This map will store all boxes for a given frame index
            frame_to_box_map = defaultdict(list)
            
            # 'item["annotations"]' is a list of tracks (for re-appearance)
            for track_id, track in enumerate(item['annotations']):
                # 'track["bboxes"]' is a list of boxes in that track
                for bbox_info in track['bboxes']:
                    frame_idx = bbox_info['frame']
                    box = [
                        bbox_info['x1'], bbox_info['y1'],
                        bbox_info['x2'], bbox_info['y2']
                    ]
                    # We add the track_id, which the original loss function expects
                    frame_to_box_map[frame_idx].append({
                        'track_id': track_id, 
                        'box': box
                    })
            
            self.annos_dict[video_id] = frame_to_box_map
        print(f"Loaded and processed {len(self.annos_dict)} video annotations.")

        # 4. Define the transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        # 1. Get the sample info
        sample_name = self.sample_names[idx]
        sample_path = os.path.join(self.samples_dir, sample_name)
        
        # --- 2. Load Query RGB Image (Randomly selected) ---
        img_idx = random.randint(1, 3)
        img_name = f"img_{img_idx}.jpg"
        img_path = os.path.join(sample_path, 'object_images', img_name)
        
        try:
            query_img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Query image not found at {img_path}")
            return None # Skip this sample
            
        query_img_tensor = self.transform(query_img)
        
        # --- 3. Get Video Info & Sampled Indices ---
        video_path = os.path.join(sample_path, 'drone_video.mp4')
        try:
            cap = cv2.VideoCapture(video_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception as e:
            print(f"Error reading video info: {video_path} | {e}")
            return None

        if total_video_frames <= 0:
            print(f"Video file has no frames: {video_path}")
            return None

        sampling_rate = total_video_frames / self.num_frames
        sampled_idxs = [int(round(sampling_rate * i)) for i in range(self.num_frames)]
        sampled_idxs = [min(max(0, i), total_video_frames - 1) for i in sampled_idxs]
        sampled_idxs = [101,102,103,1811,1812]
        
        # --- 4. Load Video Frames ---
        video_pil_frames = load_video_frames_at_indices(video_path, sampled_idxs)
        video_tensor = torch.stack(
            [self.transform(frame) for frame in video_pil_frames], 
            dim=0
        )
        
        # --- 5. Load Targets (Using our new annos_dict) ---
        if sample_name not in self.annos_dict:
            # This can happen if some samples in the 'samples' dir 
            # don't have an entry in 'annotations.json'
            print(f"Warning: No annotation entry found for video_id: {sample_name}")
            return None
            
        gt_lookup = self.annos_dict[sample_name] # This is our {frame_idx: [boxes]} map
        
        targets = dict()
        # This 'bboxes' dict format is required by the original SVOL loss
        bboxes = defaultdict(list)
        num_boxes_per_frame = [0] * self.num_frames
        
        for i, frame_idx in enumerate(sampled_idxs):
            frame_boxes = [] # Temp list for boxes in this *one* frame
            
            # Check if this exact frame has a ground truth box
            if frame_idx in gt_lookup:
                # gt_lookup[frame_idx] is a list of gt boxes, 
                # e.g. [{'track_id': 0, 'box': [...]}, {'track_id': 1, 'box': [...]}]
                for gt_obj in gt_lookup[frame_idx]:
                    xyxy_box = torch.tensor(gt_obj['box'], dtype=torch.float32)
                    
                    # Clamp box coordinates to be within image dimensions
                    xyxy_box[0::2] = torch.clamp(xyxy_box[0::2], 0, w)
                    xyxy_box[1::2] = torch.clamp(xyxy_box[1::2], 0, h)
                    
                    cxcywh_box = box_xyxy_to_cxcywh(xyxy_box)
                    
                    # Normalize
                    norm_box = cxcywh_box / torch.tensor([w, h, w, h], dtype=torch.float32)
                    
                    frame_boxes.append({
                        'track_id': gt_obj['track_id'],
                        'bbox': norm_box
                    })
            
            # Store the boxes for this frame (even if empty)
            # The key is the frame index, which is what the original loss expects
            bboxes[frame_idx] = frame_boxes
            num_boxes_per_frame[i] = len(frame_boxes)

        total_boxes = sum(num_boxes_per_frame)
        
        # We now KEEP samples with 0 boxes, as the model
        # must learn to output "no-object"
        
        # --- 6. Prepare Model Inputs and Targets ---
        model_inputs = dict()
        model_inputs['input_query_image'] = query_img_tensor.unsqueeze(0) # (1, C, H, W)
        model_inputs['input_video'] = video_tensor                 # (T, C, H, W)
        
        targets['video'] = sample_name
        targets['size'] = [w, h]
        targets['total_boxes'] = total_boxes
        targets['num_boxes_per_frame'] = num_boxes_per_frame
        targets['bboxes'] = bboxes # The dict of {frame_idx: [boxes]}
        
        return dict(model_inputs=model_inputs, targets=targets)
