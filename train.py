import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os
from tqdm import tqdm

# --- Imports from our other files ---
from model import build_model
from dataset import ZaloAIDataset
from collate_fn import custom_collate_fn, prepare_batch_inputs
# ---
# 1. UTILITY & LOSS FUNCTIONS (Copied from original repo)
# ---

# From lib.utils.box_utils
@torch.no_grad()
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

@torch.no_grad()
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    boxes1, boxes2: [N, 4] in (x0, y0, x1, y1) format
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

# From lib.utils.model_utils
@torch.no_grad()
def accuracy(output, target):
    pred = output.argmax(-1)
    correct = pred.eq(target)
    return 100 * correct.float().mean()


# From lib.modeling.matcher
class PerFrameMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 num_frames: int = 32, num_queries_per_frame: int = 10):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_queries_per_frame = num_queries_per_frame
        self.foreground_label = 0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        assert num_queries == self.num_frames * self.num_queries_per_frame

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B * N_q, 2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B * N_q, 4]

        tgt_bbox = []
        num_boxes = [] # List of num boxes per frame, flattened across batch
        for tgt_video in targets:
            num_boxes.extend(tgt_video['num_boxes_per_frame'])
            # Iterate in the order of sampled frames
            for frame_idx in tgt_video['sampled_idxs']:
                for tgt_instance in tgt_video['bboxes'][frame_idx]:
                    tgt_bbox.append(tgt_instance['bbox'])

        if not tgt_bbox:
            # No GT boxes in this batch
            return [(torch.as_tensor([], dtype=torch.long), torch.as_tensor([], dtype=torch.long)) for _ in range(bs)]

        tgt_bbox = torch.stack(tgt_bbox).to(out_bbox.device) # [total_boxes, 4]
        tgt_ids = torch.full([tgt_bbox.shape[0]], self.foreground_label, device=out_bbox.device)

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class
        C = C.view(bs * self.num_frames, self.num_queries_per_frame, -1).cpu()

        cum_num_boxes = np.cumsum(num_boxes)
        tgt_offsets = [0] + list(cum_num_boxes[:-1])
        indices = []
        
        # C shape is (B*T, 10, total_boxes). We split along dim 2
        frame_costs = C.split(num_boxes, -1)
        
        for i, (c, o) in enumerate(zip(frame_costs, tgt_offsets)):
            # c[i] shape is (10, num_boxes_for_this_frame)
            if c[i].shape[1] == 0: # No boxes for this frame
                indices.append((np.array([]), np.array([])))
                continue
            pred_ind, tgt_ind = linear_sum_assignment(c[i])
            pred_ind += i * self.num_queries_per_frame
            tgt_ind += o
            indices.append((pred_ind, tgt_ind))

        # Re-batch the indices
        indices = [indices[i:i+self.num_frames] for i in range(0, len(indices), self.num_frames)]
        
        indices_per_video = []
        for batch_idx, frame_indices in enumerate(indices):
            pred_indices_per_video = []
            tgt_indices_per_video = []
            
            # This logic is complex, it flattens all matched indices
            # for the entire video.
            base_pred_offset = batch_idx * num_queries
            base_tgt_offset = sum(sum(t['num_boxes_per_frame'] for t in targets[:batch_idx]))

            for pred_ind, tgt_ind in frame_indices:
                pred_indices_per_video.extend(pred_ind.tolist())
                tgt_indices_per_video.extend(tgt_ind.tolist())

            # De-offset them to be relative to the video
            pred_indices_per_video = [p - base_pred_offset for p in pred_indices_per_video]
            tgt_indices_per_video = [t - base_tgt_offset for t in tgt_indices_per_video]

            indices_per_video.append((
                torch.as_tensor(pred_indices_per_video, dtype=torch.int64), 
                torch.as_tensor(tgt_indices_per_video, dtype=torch.int64)
            ))
            
        return indices_per_video

def build_matcher(args):
    return PerFrameMatcher(
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class,
        num_frames=args.num_frames,
        num_queries_per_frame=args.num_queries_per_frame
    )

# From lib.modeling.loss
class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.foreground_label = 0
        self.background_label = 1
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, log=True):
        src_logits = outputs['pred_logits'] # (B, N_q, 2)
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = self.foreground_label
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes[idx])
        return losses

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] # (total_matched_boxes, 4)
        
        tgt_boxes = []
        for tgt_video in targets:
            for frame_idx in tgt_video['sampled_idxs']:
                for tgt_instance in tgt_video['bboxes'][frame_idx]:
                    tgt_boxes.append(tgt_instance['bbox'])
        
        if not tgt_boxes:
             return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device), 
                     'loss_giou': torch.tensor(0.0, device=src_boxes.device)}

        tgt_boxes = torch.stack(tgt_boxes).to(src_boxes.device)
        
        # The matcher provides indices into the *flattened* target boxes
        tgt_perm_idx = torch.cat([tgt for (_, tgt) in indices])
        tgt_boxes = tgt_boxes[tgt_perm_idx]

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)
        ))
        
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / len(indices) # Normalize by batch size
        losses['loss_giou'] = loss_giou.sum() / len(indices)
        return losses

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

def build_loss(args):
    matcher = build_matcher(args)
    weight_dict = {"loss_bbox": args.set_cost_bbox,
                   "loss_giou": args.set_cost_giou,
                   "loss_label": args.set_cost_class}
    
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes']
    return SetCriterion(matcher, weight_dict, args.eos_coef, losses)


# ---
# 2. METRIC CALCULATION (stIoU)
# ---

@torch.no_grad()
def calculate_st_iou_batch(pred_logits, pred_boxes, targets, args):
    """
    Calculates the cumulative intersection and union for stIoU for a batch.
    
    pred_logits: [N, 320, 2]
    pred_boxes: [N, 320, 4] (cxcywh)
    targets: list[dict]
    """
    batch_size = pred_logits.shape[0]
    num_frames = args.num_frames
    num_queries_per_frame = args.num_queries_per_frame
    
    # Reshape predictions to be per-frame
    # (N, 32, 10, 2)
    pred_logits = pred_logits.view(batch_size, num_frames, num_queries_per_frame, 2)
    # (N, 32, 10, 4)
    pred_boxes = pred_boxes.view(batch_size, num_frames, num_queries_per_frame, 4)
    
    # Find the best query for each frame
    # We find the query with the highest "object" score (class 0)
    # best_query_idx shape: (N, 32)
    best_query_idx = pred_logits[..., 0].argmax(dim=-1)
    
    # Gather the best box for each frame
    # We need to expand best_query_idx to match the box dimensions
    # (N, 32, 1, 1) -> (N, 32, 1, 4)
    idx_tensor = best_query_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 4)
    # (N, 32, 4)
    best_pred_boxes = pred_boxes.gather(2, idx_tensor).squeeze(2)
    
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
                # Take the first GT box if multiple exist
                gt_boxes_list.append(frame_bboxes[0]['bbox'])
            else:
                # Use a zero-box if no object in frame
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
        
        # Add to batch totals
        batch_intersection += inter.sum()
        batch_union += union.sum()
        
    return batch_intersection, batch_union

# ---
# 3. TRAIN & VALIDATE FUNCTIONS
# ---

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, args):
    model.train()
    criterion.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]")
    for batched_inputs, batched_targets in pbar:
        if batched_inputs is None:
            continue

        # 1. Prepare inputs (moves to device)
        model_inputs = prepare_batch_inputs(batched_inputs, device)
        
        # 2. Prepare targets (move bboxes to device)
        device_targets = []
        for t in batched_targets:
            dev_t = {
                'num_boxes_per_frame': t['num_boxes_per_frame'],
                'bboxes': {},
                'sampled_idxs': t['sampled_idxs'] # We need this for the matcher
            }
            for frame_idx, boxes in t['bboxes'].items():
                dev_t['bboxes'][frame_idx] = [
                    {'bbox': b['bbox'].to(device)} for b in boxes
                ]
            device_targets.append(dev_t)

        # 3. Forward pass
        outputs = model(**model_inputs)
        
        # 4. Calculate loss
        loss_dict = criterion(outputs, device_targets)
        
        # 5. Calculate total weighted loss
        total_loss_batch = sum(
            loss_dict[k] * criterion.weight_dict[k] 
            for k in loss_dict.keys() if k in criterion.weight_dict
        )

        # 6. Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch} [Training] Avg Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, data_loader, device, args):
    model.eval()
    
    total_intersection = 0.0
    total_union = 0.0
    
    pbar = tqdm(data_loader, desc="Validation")
    for batched_inputs, batched_targets in pbar:
        if batched_inputs is None:
            continue

        model_inputs = prepare_batch_inputs(batched_inputs, device)
        
        # We need device_targets for stIoU calculation
        device_targets = []
        for t in batched_targets:
            dev_t = {
                'bboxes': {},
                'sampled_idxs': t['sampled_idxs']
            }
            for frame_idx, boxes in t['bboxes'].items():
                dev_t['bboxes'][frame_idx] = [
                    {'bbox': b['bbox'].to(device)} for b in boxes
                ]
            device_targets.append(dev_t)

        # Forward pass
        outputs = model(**model_inputs)
        
        # Calculate stIoU
        inter, union = calculate_st_iou_batch(
            outputs['pred_logits'], 
            outputs['pred_boxes'], 
            device_targets, 
            args
        )
        total_intersection += inter
        total_union += union
        
        if total_union > 0:
            current_st_iou = total_intersection / total_union
            pbar.set_postfix(stIoU=f"{current_st_iou:.4f}")

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
        KAGGLE_ROOT = "/kaggle/input/zaloai"
        TRAIN_ROOT = os.path.join(KAGGLE_ROOT, "train")
        CHECKPOINT_DIR = "./checkpoints"

        # --- Data ---
        NUM_FRAMES = 32
        BATCH_SIZE = 8
        NUM_WORKERS = 2
        
        # --- Training ---
        NUM_EPOCHS = 10  # Changed to 10
        LR = 1e-4
        WD = 1e-4
        LR_DROP_STEP = 8 # Drop LR every 8 epochs (e.g., at epoch 8)
        
        # --- Model Config (from model.py) ---
        backbone = 'resnet'
        sketch_head = 'svanet'
        hidden_dim = 256
        nheads = 8
        num_layers = 4
        dim_feedforward = 1024
        num_queries = 320 # 32 frames * 10 queries/frame
        num_input_frames = 32 * 7 * 7
        num_input_sketches = 1
        input_dropout = 0.4
        n_input_proj = 2
        dropout = 0.1
        pre_norm = False
        use_sketch_pos = True
        sketch_position_embedding = 'sine'
        video_position_embedding = 'sine'
        aux_loss = True
        vis_mode = None
        input_vid_dim = None # Set by build_backbone
        input_skch_dim = None # Set by build_backbone

        # --- Loss Config ---
        matcher = 'per_frame_matcher'
        num_queries_per_frame = 10
        set_cost_bbox = 5
        set_cost_giou = 1
        set_cost_class = 2
        eos_coef = 0.1

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
    model = build_model(args).to(device)
    criterion = build_loss(args).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.WD)
    scheduler = StepLR(optimizer, step_size=args.LR_DROP_STEP, gamma=0.1)

    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    
    for epoch in range(1, args.NUM_EPOCHS + 1):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args)
        scheduler.step()
        
        print(f"--- Epoch {epoch} complete ---")

    # --- 5. Save Final Model ---
    final_checkpoint_path = os.path.join(args.CHECKPOINT_DIR, "final_model_epoch10.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training finished. Final model saved to {final_checkpoint_path}")


if __name__ == "__main__":
    main()
