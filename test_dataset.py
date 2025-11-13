import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from dataset import ZaloAIDataset  

# Cấu hình hiển thị
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def denormalize_image(tensor):
    """Chuyển Tensor (đã chuẩn hóa ImageNet) về ảnh numpy (0-255) để hiển thị"""
    # Tensor: (C, H, W) -> Numpy: (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    # Reverse normalization: img * std + mean
    img = (img * STD + MEAN) * 255
    # Clip giá trị về [0, 255] và chuyển sang uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def denormalize_bbox(norm_box, w, h):
    """
    Chuyển đổi box từ (cx, cy, w_norm, h_norm) dạng 0-1
    Về lại (x1, y1, x2, y2) dạng pixel
    """
    cx, cy, bw, bh = norm_box
    
    # 1. Nhân lại với kích thước ảnh
    cx *= w
    cy *= h
    bw *= w
    bh *= h
    
    # 2. Chuyển từ center về corners (x1, y1, x2, y2)
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)
    
    return x1, y1, x2, y2

def visualize_sample(sample):
    if sample is None:
        print("Sample is None (Loaded failed or filtered)")
        return

    inputs = sample['model_inputs']
    targets = sample['targets']
    
    # 1. Lấy Query Image
    # input_query_image shape: (1, 3, 224, 224) -> lấy phần tử đầu tiên
    query_img = denormalize_image(inputs['input_query_image'][0])
    
    # 2. Lấy Video Frames
    # input_video shape: (32, 3, 224, 224)
    video_tensor = inputs['input_video']
    
    # Thông tin kích thước gốc để vẽ box
    orig_w, orig_h = targets['size']
    # Lưu ý: Ảnh đã bị resize về 224x224, nên ta vẽ box trên nền 224x224
    # Model output box theo tỷ lệ 0-1, nên ta nhân với 224 để vẽ check
    display_w, display_h = 224, 224

    # --- VẼ MATPLOTLIB ---
    fig = plt.figure(figsize=(15, 8))
    
    # Hiển thị Query Image
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(query_img)
    ax1.set_title(f"Query: {targets['category']}")
    ax1.axis('off')
    
    # Hiển thị 7 frames đầu tiên trong chuỗi video
    sampled_idxs = targets['sampled_idxs']
    bboxes_dict = targets['bboxes']
    
    for i in range(7):
        if i >= len(video_tensor): break
        
        # Lấy frame i
        frame_img = denormalize_image(video_tensor[i])
        frame_img = frame_img.copy() # Copy để vẽ đè lên
        
        # Tìm box tương ứng:
        # Dataset trả về 1 chuỗi tensor video (index 0, 1, 2...)
        # Nhưng targets['bboxes'] lại lưu theo index gốc (vd: 3483, 3484...)
        # Ta cần map qua 'sampled_idxs'
        original_frame_idx = sampled_idxs[i]
        
        # Kiểm tra xem frame gốc này có annotation không
        if original_frame_idx in bboxes_dict:
            boxes_list = bboxes_dict[original_frame_idx]
            for obj in boxes_list:
                # obj['bbox'] đang là tensor [cx, cy, w, h] chuẩn hóa 0-1
                norm_box = obj['bbox'].numpy()
                
                # Chuyển về pixel trên ảnh 224x224
                x1, y1, x2, y2 = denormalize_bbox(norm_box, display_w, display_h)
                
                # Vẽ hình chữ nhật (Màu đỏ)
                cv2.rectangle(frame_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # Vẽ tâm (Màu xanh lá)
                cv2.circle(frame_img, (int(norm_box[0]*display_w), int(norm_box[1]*display_h)), 2, (0, 255, 0), -1)

        ax = fig.add_subplot(2, 4, i + 2)
        ax.imshow(frame_img)
        ax.set_title(f"Frame {original_frame_idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Video ID: {targets['video']}")
    print(f"Track IDs: {targets['track_ids']}")
    print(f"Original Size: {targets['size']}")
    print("-" * 20)

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Cấu hình đường dẫn (Sửa lại cho đúng máy bạn)
    ROOT_DIR = "/kaggle/input/zaloai/train" # <--- SỬA ĐƯỜNG DẪN NÀY
    
    # Khởi tạo dataset
    dataset = ZaloAIDataset(root_dir=ROOT_DIR, phase='train', num_frames=32)
    
    # Lấy ngẫu nhiên 3 mẫu để check
    for i in range(3):
        idx = np.random.randint(0, len(dataset))
        print(f"Checking sample index: {idx}")
        sample = dataset[idx]
        visualize_sample(sample)