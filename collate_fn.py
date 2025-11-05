from tensor_utils import pad_sequences_1d
import torch
def custom_collate_fn(batch):
    # Filter out any 'None' samples
    batch = [d for d in batch if d is not None]
    if not batch:
        return None, None # Return None if the whole batch was bad

    batched_targets = [d['targets'] for d in batch]
    input_keys = batch[0]['model_inputs'].keys()
    batched_inputs = dict()
    
    for k in input_keys:
        batched_inputs[k] = pad_sequences_1d(
            [d['model_inputs'][k] for d in batch],
            dtype=torch.float32,
        )
    return batched_inputs, batched_targets