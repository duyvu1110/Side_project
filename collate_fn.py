from tensor_utils import pad_sequences_1d
import torch
def custom_collate_fn(batch):
    # Filter out any 'None' samples
    batch = [d for d in batch if d is not None]
    if not batch:
        return None, None 

    # --- MODIFY THIS LINE ---
    # We only want to pad the 'model_inputs', not the 'targets'
    batched_targets = [d['targets'] for d in batch]
    
    input_keys = batch[0]['model_inputs'].keys()
    batched_inputs = dict()
    
    for k in input_keys:
        padded_data, mask = pad_sequences_1d(
            [d['model_inputs'][k] for d in batch],
            dtype=torch.float32,
            device=torch.device('cpu') 
        )
        batched_inputs[k] = (padded_data, mask)
        
    return batched_inputs, batched_targets


def prepare_batch_inputs(batched_inputs, device, non_blocking=False):
    """
    Moves the padded batch to the device and renames keys
    for the model's forward pass.
    """
    model_inputs = dict(
        # Rename 'input_query_image' to 'src_sketch' for the model
        src_sketch = batched_inputs['input_query_image'][0].to(device, non_blocking=non_blocking),
        src_sketch_mask = batched_inputs['input_query_image'][1].to(device, non_blocking=non_blocking),
        
        src_video = batched_inputs['input_video'][0].to(device, non_blocking=non_blocking),
        src_video_mask = batched_inputs['input_video'][1].to(device, non_blocking=non_blocking)
    )
    return model_inputs