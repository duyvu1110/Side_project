import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm

# --- Imports from our other files ---
from model import ZaloTrackerNet # <-- Our ResNet-34 model
from dataset import ZaloAIDataset
from collate_fn import custom_collate_fn # <-- REMOVED 'prepare_batch_inputs'
from box_utils import box_cxcywh_to_xyxy, generalized_box_iou

# ---
# 1. NEW LOSS FUNCTIONS
# ---

class FocalLoss(nn.Module):
    """
    A simple implementation of Focal Loss for binary classification.
    Assumes logits as input (not sigmoid!) and class indices (0 or 1).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs shape: [N, 2, ...] (logits for 0 and 1)
        # targets shape: [N, ...] (long tensor of 0 or 1)
        
        # Get probabilities
        p = torch.softmax(inputs, dim=1)
        
        # Gather probabilities for the target class
        # targets.unsqueeze(1) -> [N, 1, ...]
        # p_t -> [N, ...]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Get cross-entropy loss
        # We need to one-hot encode targets for binary_cross_entropy_with_logits
        # and permute to [N, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        if targets_one_hot.dim() > 2: # Handle 2D grid
             targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
             
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets_one_hot,
            reduction='none'
        ).sum(dim=1) # Sum over the class dimension
        
        # Calculate alpha weight
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        # Calculate focal loss
        loss = alpha_t * torch.pow(1 - p_t, self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def compute_losses(outputs, targets, loss_weights, grid_size=7):
    """
    Computes the dense losses for the ZaloTrackerNet.
    
    Args:
        outputs (dict): From model, contains:
            'pred_logits': [B, T, 2, 7, 7]
            'pred_boxes': [B, T, 4, 7, 7]
        targets (list[dict]): The batch of targets from the dataloader.
        loss_weights (dict): Weights for each loss type.
        grid_size (int): The size of the prediction grid (e.g., 7).
    """
    
    pred_logits = outputs['pred_logits'] # [B, T, 2, 7, 7]
    pred_boxes = outputs['pred_boxes']   # [B, T, 4, 7, 7]
    
    B, T, _, H_grid, W_grid = pred_logits.shape
    device = pred_logits.device
    
    # --- 1. Create Ground Truth Maps ---
    # channel 0 = background, channel 1 = foreground
    gt_cls_map = torch.zeros(B, T, H_grid, W_grid, dtype=torch.long, device=device)
    gt_box_map = torch.zeros(B, T, 4, H_grid, W_grid, dtype=torch.float, device=device)
    # Mask to indicate which cells have a GT box (for regression loss)
    regression_mask = torch.zeros(B, T, H_grid, W_grid, dtype=torch.bool, device=device)
    
    total_gt_boxes = 0
    
    for b in range(B):
        target = targets[b]
        for t, frame_idx in enumerate(target['sampled_idxs']):
            frame_bboxes = target['bboxes'][frame_idx]
            
            for gt_obj in frame_bboxes:
                # gt_box is [cx, cy, w, h] normalized
                gt_box = gt_obj['bbox'].to(device)
                
                # Find which grid cell the center falls into
                cx_norm, cy_norm = gt_box[0], gt_box[1]
                
                # Clamp to avoid edge cases
                cell_x = (cx_norm * W_grid).clamp(0, W_grid - 1).long()
                cell_y = (cy_norm * H_grid).clamp(0, H_grid - 1).long()
                
                # Assign to maps
                gt_cls_map[b, t, cell_y, cell_x] = 1 # 1 = foreground class
                gt_box_map[b, t, :, cell_y, cell_x] = gt_box
                regression_mask[b, t, cell_y, cell_x] = True
                total_gt_boxes += 1
    
    if total_gt_boxes == 0:
        total_gt_boxes = 1.0 # Avoid division by zero
        
    # --- 2. Calculate Classification Loss (Focal Loss) ---
    # We use all B*T*H*W samples
    
    # Flatten inputs for loss:
    # [B, T, 2, H, W] -> [B*T*H*W, 2]
    # flat_logits = pred_logits.permute(0, 1, 3, 4, 2).contiguous().view(-1, 2)
    flat_logits = pred_logits.permute(0, 1, 3, 4, 2).reshape(-1, 2)
    # [B, T, H, W] -> [B*T*H*W]
    # flat_gt_cls = gt_cls_map.view(-1)
    flat_gt_cls = gt_cls_map.reshape(-1)

    
    focal_loss_fn = FocalLoss()
    loss_cls = focal_loss_fn(flat_logits, flat_gt_cls)
    
    # --- 3. Calculate Regression Losses (L1 + GIoU) ---
    # Only use cells that have a GT box
    
    # [B, T, 4, H, W] -> [B, T, H, W, 4]
    pred_boxes_permuted = pred_boxes.permute(0, 1, 3, 4, 2).contiguous()
    gt_box_map_permuted = gt_box_map.permute(0, 1, 3, 4, 2).contiguous()

    # Select only the predictions and targets from positive cells
    # Shape of both will be [num_gt_boxes, 4]
    pred_boxes_pos = pred_boxes_permuted[regression_mask]
    gt_boxes_pos = gt_box_map_permuted[regression_mask]

    loss_bbox = torch.tensor(0.0, device=device)
    loss_giou = torch.tensor(0.0, device=device)

    if pred_boxes_pos.shape[0] > 0:
        # L1 Loss
        loss_bbox = F.l1_loss(pred_boxes_pos, gt_boxes_pos, reduction='sum') / total_gt_boxes
        
        # GIoU Loss
        # generalized_box_iou expects xyxy
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_pos)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_pos)
        
        giou = torch.diag(generalized_box_iou(pred_xyxy, gt_xyxy))
        loss_giou = (1 - giou).sum() / total_gt_boxes
        
    # --- 4. Combine Losses ---
    losses = {
        'loss_cls': loss_cls,
        'loss_bbox': loss_bbox,
        'loss_giou': loss_giou
    }
    
    total_loss = (
        loss_cls * loss_weights['loss_cls'] +
        loss_bbox * loss_weights['loss_bbox'] +
        loss_giou * loss_weights['loss_giou']
    )
    
    return total_loss, losses, total_gt_boxes

# ---
# 2. METRIC CALCULATION (stIoU)
# ---

@torch.no_grad()
def calculate_st_iou_batch(pred_logits, pred_boxes, targets, args):
    """
    Calculates stIoU for the grid-based model output.
    
    pred_logits: [N, T, 2, 7, 7]
    pred_boxes: [N, T, 4, 7, 7]
    targets: list[dict]
    """
    batch_size = pred_logits.shape[0]
    num_frames = args.NUM_FRAMES
    grid_size = args.GRID_SIZE
    
    # Get foreground scores (class 1 is foreground)
    # [N, T, 7, 7]
    foreground_scores = pred_logits[:, :, 1, :, :] # <-- **FIXED**: Use channel 1
    
    # Flatten grid to find best prediction
    # [N, T, 49]
    foreground_scores_flat = foreground_scores.view(batch_size, num_frames, -1)
    # [N, T]
    best_query_idx = foreground_scores_flat.argmax(dim=-1)
    
    # Reshape boxes to match
    # [N, T, 4, 7, 7] -> [N, T, 4, 49] -> [N, T, 49, 4]
    pred_boxes_flat = pred_boxes.flatten(3).permute(0, 1, 3, 2).contiguous()
    
    # Gather the best box for each frame
    # (N, T, 1, 1) -> (N, T, 1, 4)
    idx_tensor = best_query_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 4)
    # (N, T, 4)
    best_pred_boxes = pred_boxes_flat.gather(2, idx_tensor).squeeze(2)
    
    batch_intersection = 0.0
    batch_union = 0.0
    
    for i in range(batch_size):
        # Get per-frame predictions (xyxy)
        # (32, 4)
        pred_xyxy = box_cxcywh_to_xyxy(best_pred_boxes[i])
        
        # Get per-frame GT boxes (xyxy)
        target = targets[i]
        gt_boxes_list = []
        for frame_idx in target['sampled_idxs']:
            frame_bboxes = target['bboxes'][frame_idx]
            if frame_bboxes:
                gt_boxes_list.append(frame_bboxes[0]['bbox'])
            else:
                gt_boxes_list.append(torch.zeros(4, device=pred_xyxy.device))
        
        gt_cxcywh = torch.stack(gt_boxes_list)
        gt_xyxy = box_cxcywh_to_xyxy(gt_cxcywh)
        
        # Calculate IoU for all 32 frames
        inter_tl = torch.max(pred_xyxy[:, :2], gt_xyxy[:, :2])
        inter_br = torch.min(pred_xyxy[:, 2:], gt_xyxy[:, 2:])
        inter_wh = (inter_br - inter_tl).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1] # (32,)
        
        area_pred = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_gt = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
        union = area_pred + area_gt - inter # (32,)
        
        batch_intersection += inter.sum()
        batch_union += union.sum()
        
    return batch_intersection, batch_union

# ---
# 3. TRAIN & VALIDATE FUNCTIONS
# ---

def train_one_epoch(model, loss_weights, data_loader, optimizer, device, epoch, args):
    model.train()
    
    avg_loss_dict = defaultdict(float)
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]")
    for batched_inputs, batched_targets in pbar:
        if batched_inputs is None:
            continue
            
        optimizer.zero_grad(set_to_none=True)

        # ---
        # ** THE FIX IS HERE **
        # We manually create the model_inputs dict with *only* the keys
        # our ZaloTrackerNet model accepts.
        # This removes the 'src_sketch' argument.
        # ---
        try:
            query_images = torch.stack(batched_inputs['input_query_image']).to(device)
        except TypeError:
            query_images = batched_inputs['input_query_image'].to(device)
        model_inputs = {
            'input_query_image': batched_inputs['input_query_image'].to(device),
            'input_video': batched_inputs['input_video'].to(device)
        }
        
        # We also need to move target boxes to device, but we'll do it
        # inside the compute_losses function as we build the maps.
        
        outputs = model(**model_inputs)
        
        total_loss, loss_dict, num_gts = compute_losses(
            outputs, 
            batched_targets, 
            loss_weights, 
            grid_size=args.GRID_SIZE
        )
        
        if not torch.isfinite(total_loss):
            print(f"Warning: NaN or Inf loss detected at epoch {epoch}. Skipping batch.")
            continue
        
        total_loss.backward()
        optimizer.step()
        
        num_batches += 1
        avg_loss_dict['total_loss'] += total_loss.item()
        avg_loss_dict['gt_boxes'] += num_gts
        
        for k, v in loss_dict.items():
            avg_loss_dict[k] += v.item()

        if num_batches > 0:
            pbar.set_postfix(
                loss=f"{avg_loss_dict['total_loss'] / num_batches:.4f}",
                gt_boxes=f"{int(avg_loss_dict['gt_boxes'] / num_batches)}"
            )

    if num_batches > 0:
        final_loss_dict = {k: v / num_batches for k, v in avg_loss_dict.items()}
        return final_loss_dict
    else:
        print(f"Epoch {epoch} [Training] No valid batches found.")
        return None


@torch.no_grad()
def validate(model, data_loader, device, args):
    model.eval()
    
    total_intersection = 0.0
    total_union = 0.0
    
    pbar = tqdm(data_loader, desc="Validation")
    for batched_inputs, batched_targets in pbar:
        if batched_inputs is None:
            continue

        # ---
        # ** THE FIX IS HERE **
        # Apply the same fix as in the training loop.
        # ---
        model_inputs = {
            'input_query_image': batched_inputs['input_query_image'].to(device),
            'input_video': batched_inputs['input_video'].to(device)
        }
        
        outputs = model(**model_inputs)
        
        inter, union = calculate_st_iou_batch(
            outputs['pred_logits'], 
            outputs['pred_boxes'], 
            batched_targets, # This function handles moving GTs to device
            args
        )
        total_intersection += inter
        total_union += union
        
        if total_union > 0:
            current_st_iou = total_intersection / total_union
            pbar.set_postfix(stIoU=f"{current_st_iou:.4f}")
    
    if total_union == 0:
        print("Validation: No objects found or predicted, stIoU is 0.")
        return 0.0

    final_st_iou = total_intersection / total_union
    print(f"Validation stIoU: {final_st_iou:.4f}")
    return final_st_iou

# ---
# 4. MAIN FUNCTION
# ---
def main():
    
    # --- 1. Configuration ---
    class Config:
        # --- Paths ---
        KAGGLE_ROOT = "/kaggle/input/zaloai" # <-- Using a local path for testing
        TRAIN_ROOT = os.path.join(KAGGLE_ROOT, "train")
        CHECKPOINT_DIR = "./checkpoints"

        # --- Data ---
        NUM_FRAMES = 32
        BATCH_SIZE = 8 # Can try a larger batch size now
        NUM_WORKERS = 4
        
        # --- Training ---
        NUM_EPOCHS = 100
        LR = 1e-4
        WD = 1e-4
        LR_DROP_STEP = 80
        
        # --- Model Config ---
        GRID_SIZE = 7 # From ResNet-34 layer4
        
        # --- Loss Weights ---
        loss_weight_cls = 1.0   # Focal loss is good at balancing, 1.0 is fine
        loss_weight_bbox = 5.0  # L1 loss
        loss_weight_giou = 2.0  # GIoU loss

    args = Config()
    os.makedirs(args.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Setup Data ---
    print("Setting up DataLoader...")
    train_dataset = ZaloAIDataset(
        root_dir=args.TRAIN_ROOT, 
        phase='train', 
        num_frames=args.NUM_FRAMES
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
        collate_fn=custom_collate_fn, shuffle=True
    )
    print(f"Train batches: {len(train_loader)}")

    # --- 3. Setup Model, Loss, Optimizer ---
    print("Building model...")
    model = ZaloTrackerNet().to(device)
    
    # Simple dictionary for loss weights
    loss_weights = {
        "loss_cls": args.loss_weight_cls,
        "loss_bbox": args.loss_weight_bbox,
        "loss_giou": args.loss_weight_giou,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.WD)
    scheduler = StepLR(optimizer, step_size=args.LR_DROP_STEP, gamma=0.1)
    
    # --- 4. Training Loop ---
    print(f"--- Starting Training on {device} ---")
    
    for epoch in range(1, args.NUM_EPOCHS + 1):
        
        loss_dict = train_one_epoch(
            model, 
            loss_weights, 
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            args
        )
        
        if loss_dict is not None:
            print(f"Epoch {epoch} Avg Losses:")
            sorted_keys = sorted(loss_dict.keys())
            for k in sorted_keys:
                if k != 'total_loss':
                    print(f"  {k:<20}: {loss_dict[k]:.4f}")
            print(f"  ==> {'total_loss':<20}: {loss_dict['total_loss']:.4f}")
        
        # Simple validation step (can be expanded)
        if epoch % 10 == 0 or epoch == args.NUM_EPOCHS:
             # Using train_loader for validation just to check logic
             # In a real setup, you'd have a separate validation_loader
             validate(model, train_loader, device, args) 
        
        scheduler.step()
        print(f"--- Epoch {epoch} complete ---")

    # --- 5. Save Final Model ---
    final_checkpoint_path = os.path.join(args.CHECKPOINT_DIR, f"final_model_epoch{args.NUM_EPOCHS}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training finished. Final model saved to {final_checkpoint_path}")

if __name__ == "__main__":
    main()