import os
import json
import torch
import random
import cv2
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from box_utils import box_xyxy_to_cxcywh
from helper_function import load_video_frames_at_indices

class ZaloAIDataset(Dataset):
    def __init__(self, root_dir, phase='train', num_frames=32):
        super(ZaloAIDataset, self).__init__()
        
        self.root_dir = root_dir
        self.phase = phase
        self.num_frames = num_frames
        
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.annos_path = os.path.join(self.root_dir, 'annotations', 'annotations.json')
        
        # 1. Get a list of all sample folders (e.g., "Backpack_0", "Backpack_1", ...)
        self.sample_names = os.listdir(self.samples_dir)
        # Filter out any hidden files
        self.sample_names = [s for s in self.sample_names if not s.startswith('.')]
        
        # 2. Load and process the annotations
        print(f"Loading annotations from {self.annos_path}...")
        with open(self.annos_path) as f:
            all_annos = json.load(f)
            
        # 3. Convert annotations list to a fast-lookup dictionary
        # This is the most important step for performance.
        self.annos_dict = {}
        for item in all_annos:
            video_id = item['video_id']
            # Create a lookup table for {frame_idx: [list of boxes]}
            gt_lookup = defaultdict(list)
            
            # "annotations" is a list of tracks (for re-appearance)
            for track_id, track in enumerate(item['annotations']):
                for bbox_info in track['bboxes']:
                    frame_idx = bbox_info['frame']
                    box = [
                        bbox_info['x1'], bbox_info['y1'],
                        bbox_info['x2'], bbox_info['y2']
                    ]
                    gt_lookup[frame_idx].append({'track_id': track_id, 'box': box})
            
            self.annos_dict[video_id] = gt_lookup
        print(f"Loaded {len(self.annos_dict)} video annotations.")

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
        sample_name = self.sample_names[idx] # e.g., "Backpack_0"
        sample_path = os.path.join(self.samples_dir, sample_name)
        
        # --- 2. Load Query RGB Image (Randomly selected) ---
        img_idx = random.randint(1, 3)
        img_name = f"img_{img_idx}.jpg"
        img_path = os.path.join(sample_path, 'object_images', img_name)
        
        query_img = Image.open(img_path).convert('RGB')
        query_img_tensor = self.transform(query_img)
        
        # --- 3. Get Video Info & Sampled Indices ---
        video_path = os.path.join(sample_path, 'drone_video.mp4')
        cap = cv2.VideoCapture(video_path)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if total_video_frames <= 0:
            raise ValueError(f"Video file has no frames: {video_path}")

        # Create the sample indices
        sampling_rate = total_video_frames / self.num_frames
        sampled_idxs = [int(round(sampling_rate * i)) for i in range(self.num_frames)]
        # Ensure indices are within bounds
        sampled_idxs = [min(max(0, i), total_video_frames - 1) for i in sampled_idxs]
        
        # --- 4. Load Video Frames ---
        video_pil_frames = load_video_frames_at_indices(video_path, sampled_idxs)
        video_tensor = torch.stack(
            [self.transform(frame) for frame in video_pil_frames], 
            dim=0
        ) # Shape: (T, C, H, W)
        
        # --- 5. Load Targets (Matching sampled indices to annos) ---
        gt_lookup = self.annos_dict[sample_name]
        
        targets = dict()
        # This 'bboxes' dict format is required by the original SVOL loss
        bboxes = defaultdict(list)
        num_boxes_per_frame = [0] * self.num_frames
        
        for i, frame_idx in enumerate(sampled_idxs):
            frame_boxes = []
            
            # Check if this exact frame has a ground truth box
            if frame_idx in gt_lookup:
                for gt_obj in gt_lookup[frame_idx]:
                    # Zalo format is [x1, y1, x2, y2]
                    xyxy_box = torch.tensor(gt_obj['box'], dtype=torch.float32)
                    
                    # Convert to [cx, cy, w, h]
                    cxcywh_box = box_xyxy_to_cxcywh(xyxy_box)
                    
                    # Normalize
                    norm_box = cxcywh_box / torch.tensor([w, h, w, h], dtype=torch.float32)
                    
                    frame_boxes.append({
                        'track_id': gt_obj['track_id'],
                        'bbox': norm_box
                    })
            
            # Store the boxes for this frame (even if empty)
            # The key MUST be the frame index
            bboxes[frame_idx] = frame_boxes
            num_boxes_per_frame[i] = len(frame_boxes)

        total_boxes = sum(num_boxes_per_frame)
        
        # If a video has no boxes at all in the sampled frames, it's not
        # a useful training sample. We'll skip it in the collate_fn.
        if total_boxes == 0:
            # print(f"Warning: No GT boxes found for {sample_name} in sampled frames. Skipping.")
            return None # Will be filtered by collate_fn
        
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