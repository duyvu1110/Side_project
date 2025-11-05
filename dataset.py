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
        self.annos_dict = {}
        for item in all_annos_list:
            video_id = item['video_id']
            frame_to_box_map = defaultdict(list)
            
            for track_id, track in enumerate(item['annotations']):
                for bbox_info in track['bboxes']:
                    frame_idx = bbox_info['frame']
                    box = [
                        bbox_info['x1'], bbox_info['y1'],
                        bbox_info['x2'], bbox_info['y2']
                    ]
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
            # print(f"Error: Query image not found at {img_path}")
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
            # print(f"Error reading video info: {video_path} | {e}")
            return None

        if total_video_frames <= 0 or w <= 0 or h <= 0:
            # print(f"Video file has no frames or invalid dims: {video_path}")
            return None

        sampling_rate = total_video_frames / self.num_frames
        sampled_idxs = [int(round(sampling_rate * i)) for i in range(self.num_frames)]
        sampled_idxs = [min(max(0, i), total_video_frames - 1) for i in sampled_idxs]
        
        # --- 4. Load Video Frames ---
        video_pil_frames = load_video_frames_at_indices(video_path, sampled_idxs)
        video_tensor = torch.stack(
            [self.transform(frame) for frame in video_pil_frames], 
            dim=0
        )
        
        # --- 5. Load Targets (Using our new annos_dict) ---
        if sample_name not in self.annos_dict:
            # print(f"Warning: No annotation entry found for video_id: {sample_name}")
            return None
            
        gt_lookup = self.annos_dict[sample_name] 
        
        targets = dict()
        bboxes = defaultdict(list)
        num_boxes_per_frame = [0] * self.num_frames
        
        for i, frame_idx in enumerate(sampled_idxs):
            frame_boxes = []
            if frame_idx in gt_lookup:
                for gt_obj in gt_lookup[frame_idx]:
                    
                    # ---
                    # **THE FIX IS HERE**
                    # ---
                    box = gt_obj['box'] # [x1, y1, x2, y2]
                    
                    # 1. Ensure x1 < x2 and y1 < y2
                    valid_x1 = min(box[0], box[2])
                    valid_y1 = min(box[1], box[3])
                    valid_x2 = max(box[0], box[2])
                    valid_y2 = max(box[1], box[3])

                    # 2. Clamp to image boundaries
                    valid_x1 = max(0, valid_x1)
                    valid_y1 = max(0, valid_y1)
                    valid_x2 = min(w, valid_x2)
                    valid_y2 = min(h, valid_y2)
                    
                    # 3. Create the valid tensor
                    xyxy_box = torch.tensor([valid_x1, valid_y1, valid_x2, valid_y2], dtype=torch.float32)

                    # Skip zero-area boxes, which can also cause errors
                    if (xyxy_box[2] <= xyxy_box[0]) or (xyxy_box[3] <= xyxy_box[1]):
                        continue
                    
                    # --- END FIX ---
                    
                    cxcywh_box = box_xyxy_to_cxcywh(xyxy_box)
                    norm_box = cxcywh_box / torch.tensor([w, h, w, h], dtype=torch.float32)
                    
                    frame_boxes.append({
                        'track_id': gt_obj['track_id'],
                        'bbox': norm_box
                    })
            
            bboxes[frame_idx] = frame_boxes
            num_boxes_per_frame[i] = len(frame_boxes)

        total_boxes = sum(num_boxes_per_frame)
        
        # --- 6. Prepare Model Inputs and Targets ---
        model_inputs = dict()
        model_inputs['input_query_image'] = query_img_tensor.unsqueeze(0)
        model_inputs['input_video'] = video_tensor
        
        targets['video'] = sample_name
        targets['size'] = [w, h]
        targets['total_boxes'] = total_boxes
        targets['num_boxes_per_frame'] = num_boxes_per_frame
        targets['bboxes'] = bboxes
        targets['sampled_idxs'] = sampled_idxs 
        
        return dict(model_inputs=model_inputs, targets=targets)