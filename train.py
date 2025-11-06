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
from box_utils import box_cxcywh_to_xyxy, generalized_box_iou
# ---
# 1. UTILITY & LOSS FUNCTIONS (Copied from original repo)
# ---

# From lib.utils.model_utils
@torch.no_grad()
def accuracy(output, target):
    pred = output.argmax(-1)
    correct = pred.eq(target)
    return 100 * correct.float().mean()


# From lib.modeling.matcher
class PerFrameMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 num_frames: int = 32, num_queries_per_frame: int = 10):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bbox in the matching cost
            num_frames: sampled frames from video
            num_queries_per_frame: number of query slots per frame
        """
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
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted bbox coordinates
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "bboxes": Tensor of dim [num_target_bboxes, 4] containing the target bbox coordinates.
        
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_bboxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        assert num_queries == self.num_frames * self.num_queries_per_frame

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * #queries, 2]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * #queries, 4]

        tgt_bbox = []
        num_boxes = []
        for tgt_video in targets:  # batch-level
            num_boxes.extend(tgt_video['num_boxes_per_frame'])
            for tgt_frame in tgt_video['bboxes'].values():  # video-level
                for tgt_instance in tgt_frame:  # frame-level
                    tgt_bbox.append(tgt_instance['bbox'])  # instance-level

        tgt_bbox = torch.stack(tgt_bbox).to(out_bbox.device)  # [total #boxes in batch, 4]
        tgt_ids = torch.full([tgt_bbox.shape[0]], self.foreground_label)   # [total #boxes in batch]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]  # [batch_size * #queries, total #boxes in batch]

        # Compute the L1 cost between bboxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [batch_size * #queries, total #boxes in batch]

        # Compute the giou cost between bboxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))  # [batch_size * #queries, total #boxes in batch]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou + self.cost_class * cost_class  # [batch_size * #queries, total #boxes in batch]
        C = C.view(bs*self.num_frames, self.num_queries_per_frame, -1).cpu()  # [batch_size * #frames, #queries in frame, total #boxes in batch]

        # Get pairs of (pred, target)
        cum_num_boxes = np.cumsum(num_boxes)
        tgt_offsets = [0] + list(cum_num_boxes[:-1])
        indices = []
        for i, (c, o) in enumerate(zip(C.split(num_boxes, -1), tgt_offsets)):
            pred_ind, tgt_ind = linear_sum_assignment(c[i])
            pred_ind += i * self.num_queries_per_frame
            tgt_ind += o
            indices.append((pred_ind, tgt_ind))

        # yield successive #frames-sized chunks from lists
        indices = [indices[i:i+self.num_frames] for i in range(0, len(indices), self.num_frames)]  # [batch_size, #frames]

        # aggregate frame-level indices into video-level indices
        # and adjust prediction & target indices
        cum_num_queries = np.cumsum([num_queries] * bs)
        pred_offsets = [0] + list(cum_num_queries[:-1])
        indices_per_video = []
        for indices_, pred_offset in zip(indices, pred_offsets):
            pred_indices_per_video = []
            tgt_indices_per_video = []
            for indice in indices_:
                pred_ind, tgt_ind = indice
                for pred_idx, tgt_idx in zip(pred_ind, tgt_ind):
                    pred_indices_per_video.append(pred_idx - pred_offset)
                    tgt_indices_per_video.append(tgt_idx)
            tgt_indices_per_video = np.asarray(tgt_indices_per_video)
            tgt_indices_per_video = list(tgt_indices_per_video - np.min(tgt_indices_per_video))
            indices_per_video.append((pred_indices_per_video, tgt_indices_per_video))
        indices = indices_per_video

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def build_matcher(args):
    return PerFrameMatcher(
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        cost_class=args.set_cost_class,
        num_frames=args.NUM_FRAMES,
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

    def get_num_boxes(self, targets):
        """ Get the total number of GT boxes in the batch """
        num_boxes = sum(t['total_boxes'] for t in targets)
        
        # --- THIS IS THE FIX ---
        # Get device from the registered buffer 'empty_weight' instead of parameters
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=self.empty_weight.device)
        # --- END FIX ---
        
        # In a distributed setting, this would be averaged, but for one GPU, this is fine.
        return num_boxes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = self.foreground_label
        
        # ---
        # ** FIX: Normalize by num_boxes **
        # We compute the CE loss for all 320 queries, then take the sum
        # and normalize by the number of GT boxes (not the number of queries).
        # ---
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.sum() / num_boxes}

        if log:
            num_matches = idx[0].shape[0]
            if num_matches > 0:
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes[idx])
            else:
                losses['class_error'] = 0.0
                
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        
        if src_boxes.shape[0] == 0:
            return {'loss_bbox': torch.tensor(0.0, device=src_boxes.device), 
                    'loss_giou': torch.tensor(0.0, device=src_boxes.device)}
        
        tgt_boxes_list = []
        for tgt_video in targets:
            for frame_idx in tgt_video['sampled_idxs']:
                for tgt_instance in tgt_video['bboxes'][frame_idx]:
                    tgt_boxes_list.append(tgt_instance['bbox'])
        
        tgt_boxes = torch.stack(tgt_boxes_list).to(src_boxes.device)
        tgt_perm_idx = torch.cat([tgt for (_, tgt) in indices])
        tgt_boxes = tgt_boxes[tgt_perm_idx]

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)
        ))
        
        losses = {}
        # ---
        # ** FIX: Normalize by num_boxes **
        # ---
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        if loss == 'labels':
            return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        elif loss == 'boxes':
            return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        indices = self.matcher(outputs_without_aux, targets)
        
        # Get the total number of boxes (our normalization factor)
        num_boxes = self.get_num_boxes(targets)
        # Avoid division by zero if a batch has 0 boxes
        if num_boxes == 0:
             num_boxes = 1.0

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes=num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes=num_boxes, log=False)
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
    
    avg_loss_dict = defaultdict(float)
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]")
    for batched_inputs, batched_targets in pbar:
        if batched_inputs is None:
            continue
            
        optimizer.zero_grad(set_to_none=True)

        model_inputs = prepare_batch_inputs(batched_inputs, device)
        
        device_targets = []
        total_gt_boxes_in_batch = 0
        for t in batched_targets:
            dev_t = {
                'video': t['video'],
                'size': t['size'],
                'num_boxes_per_frame': t['num_boxes_per_frame'],
                'bboxes': {},
                'sampled_idxs': t['sampled_idxs'],
                'total_boxes': t['total_boxes'] # Pass this for normalization
            }
            total_gt_boxes_in_batch += t['total_boxes']
            for frame_idx, boxes in t['bboxes'].items():
                dev_t['bboxes'][frame_idx] = [
                    {'bbox': b['bbox'].to(device)} for b in boxes
                ]
            device_targets.append(dev_t)

        outputs = model(**model_inputs)
        
        # ** FIX: The criterion now handles normalization internally **
        loss_dict = criterion(outputs, device_targets)
        
        total_loss_batch = sum(
            loss_dict[k] * criterion.weight_dict[k] 
            for k in loss_dict.keys() if k in criterion.weight_dict
        )

        if not torch.isfinite(total_loss_batch):
            print(f"Warning: NaN or Inf loss detected at epoch {epoch}. Skipping batch.")
            continue

        total_loss_batch.backward()
        optimizer.step()
        
        num_batches += 1
        avg_loss_dict['total_loss'] += total_loss_batch.item()
        
        with torch.no_grad():
            indices = criterion.matcher(outputs, device_targets)
            num_matches = sum(len(i[0]) for i in indices)
            avg_loss_dict['matches'] += num_matches
            avg_loss_dict['gt_boxes'] += total_gt_boxes_in_batch

        for k, v in loss_dict.items():
            if k in criterion.weight_dict:
                avg_loss_dict[k] += v.item()

        if num_batches > 0:
            pbar.set_postfix(
                loss=f"{avg_loss_dict['total_loss'] / num_batches:.4f}",
                matches=f"{int(avg_loss_dict['matches'] / num_batches)}/{int(avg_loss_dict['gt_boxes'] / num_batches)}"
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
def nan_hook(module, inputs, output):
    """
    A simple hook function to check for NaNs in a module's output.
    """
    if isinstance(output, torch.Tensor):
        if torch.isnan(output).any():
            print(f"!!! NaN DETECTED IN OUTPUT OF: {module.__class__.__name__} !!!")
            print("--- Input (first 5 values) ---")
            if isinstance(inputs, tuple) and inputs:
                print(inputs[0].detach().cpu().flatten()[:5])
            print("--- Output (first 5 values) ---")
            print(output.detach().cpu().flatten()[:5])
            
    elif isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
             if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                print(f"!!! NaN DETECTED IN OUTPUT TENSOR {i} OF: {module.__class__.__name__} !!!")
                print("--- Output (first 5 values) ---")
                print(out.detach().cpu().flatten()[:5])

def add_nan_check_hooks(model):
    """
    Recursively apply the nan_hook to all modules in the model.
    """
    for name, module in model.named_children():
        if module is not None:
            module.register_forward_hook(nan_hook)
            add_nan_check_hooks(module) # Recurse
def main():
    
    # --- 1. Configuration ---
    class Config:
        # --- Paths ---
        KAGGLE_ROOT = "/kaggle/input/zaloai"
        TRAIN_ROOT = os.path.join(KAGGLE_ROOT, "train")
        CHECKPOINT_DIR = "./checkpoints"

        # --- Data ---
        NUM_FRAMES = 32
        BATCH_SIZE = 2
        NUM_WORKERS = 4
        
        # --- Training ---
        NUM_EPOCHS = 100
        LR = 1e-5  # We can try a higher LR now that losses are stable
        WD = 1e-4
        LR_DROP_STEP = 80
        
        # --- Model Config ---
        backbone = 'resnet'
        sketch_head = 'svanet'
        hidden_dim = 256
        nheads = 8
        num_layers = 4
        dim_feedforward = 1024
        num_queries = 320
        num_input_frames = 32 * 7 * 7
        num_input_sketches = 1
        input_dropout = 0.4
        n_input_proj = 2
        dropout = 0.1
        pre_norm = True
        use_sketch_pos = True
        sketch_position_embedding = 'sine'
        video_position_embedding = 'sine'
        aux_loss = True
        vis_mode = None
        input_vid_dim = None
        input_skch_dim = None

        # --- Loss Config ---
        matcher = 'per_frame_matcher'
        num_queries_per_frame = 10
        
        # ---
        # ** NEW WEIGHTS for new normalization **
        # ---
        set_cost_bbox = 5.0
        set_cost_giou = 2.0
        set_cost_class = 1.0 # Matcher costs are unchanged
        
        eos_coef = 0.1 # This is the weight for "no-object" class in loss_label
        
        # ---
        # ** NEW LOSS WEIGHTS for total loss **
        # ---
        loss_weight_bbox = 1.0
        loss_weight_giou = 2.0
        loss_weight_label = 1.0

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
    
    # ---
    # ** FIX: Pass new loss weights to build_loss **
    # ---
    criterion = build_loss(args).to(device)
    
    # Manually set the final loss weights in the criterion
    # (The build_loss function you have doesn't do this automatically)
    criterion.weight_dict = {
        "loss_bbox": args.loss_weight_bbox,
        "loss_giou": args.loss_weight_giou,
        "loss_label": args.loss_weight_label
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in criterion.weight_dict.items()})
        criterion.weight_dict.update(aux_weight_dict)
    # --- END FIX ---

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.WD)
    scheduler = StepLR(optimizer, step_size=args.LR_DROP_STEP, gamma=0.1)

    # --- 4. Training Loop ---
    print("--- Starting Training ---")
    
    for epoch in range(1, args.NUM_EPOCHS + 1):
        
        loss_dict = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args)
        
        if loss_dict is not None:
            print(f"Epoch {epoch} Avg Losses:")
            sorted_keys = sorted(loss_dict.keys(), key=lambda x: ('loss' not in x, 'aux' in x, x))
            for k in sorted_keys:
                if k != 'total_loss':
                    print(f"  {k:<20}: {loss_dict[k]:.4f}")
            print(f"  ==> {'total_loss':<20}: {loss_dict['total_loss']:.4f}")
        
        scheduler.step()
        print(f"--- Epoch {epoch} complete ---")

    # --- 5. Save Final Model ---
    final_checkpoint_path = os.path.join(args.CHECKPOINT_DIR, f"final_model_epoch{args.NUM_EPOCHS}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Training finished. Final model saved to {final_checkpoint_path}")

if __name__ == "__main__":
    main()
