import os
import json
import torch
import random
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision import transforms
from box_utils import box_xyxy_to_cxcywh

class ZaloAIDataset(Dataset):
    def __init__(self, root_dir, phase='train', num_frames=32):
        super(ZaloAIDataset, self).__init__()
        
        self.root_dir = root_dir
        self.phase = phase
        self.num_frames = num_frames
        
        # Đường dẫn tới thư mục chứa video và file annotations
        self.samples_dir = os.path.join(self.root_dir, 'samples')
        self.annos_path = os.path.join(self.root_dir, 'annotations', 'annotations.json')
        
        # 1. Lấy danh sách các folder mẫu (VD: Backpack_0, Lifering_1, ...)
        self.sample_names = [s for s in os.listdir(self.samples_dir) if not s.startswith('.')]
        
        # 2. Load toàn bộ annotations vào bộ nhớ và tạo dictionary tra cứu nhanh
        print(f"Loading annotations from {self.annos_path}...")
        with open(self.annos_path) as f:
            all_annos_list = json.load(f)
            
        self.annos_dict = {}
        for item in all_annos_list:
            video_id = item['video_id']
            # Tạo lookup map: { frame_index: [list of boxes] }
            frame_to_box_map = defaultdict(list)
            
            # Duyệt qua các track (đối tượng) trong video
            for track_id, track in enumerate(item['annotations']):
                for bbox_info in track['bboxes']:
                    frame_idx = bbox_info['frame']
                    # Format gốc của bạn: [x1, y1, x2, y2]
                    box = [
                        bbox_info['x1'], bbox_info['y1'],
                        bbox_info['x2'], bbox_info['y2']
                    ]
                    # Lưu kèm track_id để model hiểu đó là cùng 1 vật thể
                    frame_to_box_map[frame_idx].append({
                        'track_id': track_id, 
                        'box': box
                    })
            self.annos_dict[video_id] = frame_to_box_map
            
        # 3. Định nghĩa Transform (Chuẩn hóa theo ImageNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        # Lấy thông tin mẫu hiện tại
        sample_name = self.sample_names[idx] # VD: "Backpack_0"
        sample_path = os.path.join(self.samples_dir, sample_name)
        category = sample_name.split('_')[0] # VD: "Backpack"
        
        # --- PHẦN 1: XỬ LÝ INPUT ẢNH QUERY (Thay thế Sketch) ---
        # Lấy ngẫu nhiên 1 ảnh từ thư mục object_images
        img_idx = random.randint(1, 3)
        img_name = f"img_{img_idx}.jpg"
        img_path = os.path.join(sample_path, 'object_images', img_name)
        
        try:
            query_img = Image.open(img_path).convert('RGB')
            query_tensor = self.transform(query_img)
            # Thêm dimension batch ảo: (3, 224, 224) -> (1, 3, 224, 224)
            # Để giả lập input này giống như 1 "sketch sequence" có độ dài 1
            query_tensor = query_tensor.unsqueeze(0)
        except Exception as e:
            # print(f"Error loading image: {img_path}")
            return None

        # --- PHẦN 2: XỬ LÝ VIDEO (Lấy mẫu Frames) ---
        video_path = os.path.join(sample_path, 'drone_video.mp4')
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if total_frames <= 0: return None

        # Tính toán các frame cần lấy (chia đều video thành 32 đoạn)
        sampled_idxs = [int(round((total_frames / self.num_frames) * i)) for i in range(self.num_frames)]
        sampled_idxs = [min(max(0, i), total_frames - 1) for i in sampled_idxs]
        
        # Đọc video tại các vị trí đã chọn
        video_tensors = []
        for frame_idx in sampled_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Fallback: Dùng frame đen nếu lỗi
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame_pil = Image.fromarray(frame)
            video_tensors.append(self.transform(frame_pil))
        cap.release()
        
        # Stack lại thành tensor (T, C, H, W)
        video_tensor = torch.stack(video_tensors, dim=0)

        # --- PHẦN 3: TẠO TARGETS (Chuyển đổi format quan trọng) ---
        if sample_name not in self.annos_dict: return None
        gt_lookup = self.annos_dict[sample_name]
        
        # Cấu trúc targets bắt buộc của SVOL
        targets = {
            'video': sample_name,
            'size': [w, h],
            'category': category,
            'sampled_idxs': sampled_idxs, # Để dùng cho Matcher
            'bboxes': defaultdict(list),  # { frame_idx: [boxes] }
            'num_boxes_per_frame': []
        }
        
        all_track_ids = set()
        
        for i, frame_idx in enumerate(sampled_idxs):
            frame_boxes = []
            # Nếu frame này có annotation
            if frame_idx in gt_lookup:
                for gt_obj in gt_lookup[frame_idx]:
                    # 1. Lấy box gốc [x1, y1, x2, y2]
                    box = gt_obj['box']
                    
                    # 2. Đảm bảo tọa độ hợp lệ và nằm trong ảnh
                    valid_x1 = max(0, min(box[0], box[2]))
                    valid_y1 = max(0, min(box[1], box[3]))
                    valid_x2 = min(w, max(box[0], box[2]))
                    valid_y2 = min(h, max(box[1], box[3]))
                    
                    # 3. Tạo tensor xyxy
                    xyxy_box = torch.tensor([valid_x1, valid_y1, valid_x2, valid_y2], dtype=torch.float32)
                    
                    # 4. BƯỚC QUAN TRỌNG: Chuyển sang cxcywh và Chuẩn hóa (0-1)
                    # Model yêu cầu format này để tính loss!
                    cxcywh_box = box_xyxy_to_cxcywh(xyxy_box)
                    norm_box = cxcywh_box / torch.tensor([w, h, w, h], dtype=torch.float32)
                    
                    frame_boxes.append({
                        'track_id': gt_obj['track_id'],
                        'bbox': norm_box # Box đã chuẩn hóa
                    })
                    all_track_ids.add(gt_obj['track_id'])
            
            targets['bboxes'][frame_idx] = frame_boxes
            targets['num_boxes_per_frame'].append(len(frame_boxes))

        targets['total_boxes'] = sum(targets['num_boxes_per_frame'])
        targets['track_ids'] = list(all_track_ids)

        # --- PHẦN 4: ĐÓNG GÓI INPUT ---
        model_inputs = {
            # Đặt tên là src_sketch để khớp với model SVOL, dù ta truyền ảnh RGB
            'input_query_image': query_tensor, 
            'input_video': video_tensor
        }

        return dict(model_inputs=model_inputs, targets=targets)