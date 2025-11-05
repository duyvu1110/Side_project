import os
from dataset import ZaloAIDataset
from collate_fn import custom_collate_fn
from torch.utils.data import DataLoader
import torch


# ---
# 5. TEST SCRIPT
# ---

if __name__ == "__main__":
    
    # --- Configuration ---
    # !! SET YOUR PATHS HERE !!
    KAGGLE_ROOT = "/kaggle/input/zaloai"
    TRAIN_ROOT = os.path.join(KAGGLE_ROOT, "train")

    # Check if the path exists
    if not os.path.exists(TRAIN_ROOT):
        print(f"Error: Path not found: {TRAIN_ROOT}")
        print("Please update KAGGLE_ROOT and TRAIN_ROOT variables to point to your data.")
    else:
        print(f"Found data root: {TRAIN_ROOT}")

        NUM_FRAMES = 32
        BATCH_SIZE = 4
        NUM_WORKERS = 2 # Use 0 if you get errors, 2 is faster
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # --- 1. Create Dataset ---
        print("\n--- 1. Initializing ZaloAIDataset ---")
        try:
            dataset = ZaloAIDataset(
                root_dir=TRAIN_ROOT, 
                phase='train', 
                num_frames=NUM_FRAMES
            )
            print(f"Dataset initialized. Total samples found: {len(dataset)}")
            
            # --- 2. Create DataLoader ---
            print("\n--- 2. Initializing DataLoader ---")
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                collate_fn=custom_collate_fn,
                shuffle=True
            )

            # --- 3. Fetch One Batch ---
            print("\n--- 3. Fetching one batch ---")
            data_iter = iter(loader)
            batched_inputs, batched_targets = next(data_iter)
            
            if batched_inputs is None:
                raise ValueError("Data loader returned an empty batch. Check 'None' filtering.")

            print("Batch loaded successfully.")
            
            # --- 4. Inspect Batch Contents ---
            print("\n--- 4. Inspecting Batch ---")
            print(f"  Batched Inputs Keys: {batched_inputs.keys()}")
            print(f"  Batched Targets length (batch size): {len(batched_targets)}")
            
            # Inspect Inputs
            query_data, query_mask = batched_inputs['input_query_image']
            video_data, video_mask = batched_inputs['input_video']
            print(f"  Query Image Data Shape: {query_data.shape}")
            print(f"  Query Image Mask Shape: {query_mask.shape}")
            print(f"  Video Data Shape: {video_data.shape}")
            print(f"  Video Mask Shape: {video_mask.shape}")

            # Inspect Targets
            first_target = batched_targets[0]
            print(f"\n  First Target's Keys: {first_target.keys()}")
            print(f"  First Target's Video ID: {first_target['video']}")
            print(f"  First Target's Total Boxes (in sampled frames): {first_target['total_boxes']}")
            print(f"  First Target's bboxes dict (first 5 frames):")
            for i, (frame_idx, boxes) in enumerate(first_target['bboxes'].items()):
                if i >= 5: break
                print(f"    Frame {frame_idx}: {len(boxes)} boxes")
                if boxes:
                    print(f"      Box 0: {boxes[0]}") # Should be {'track_id': ..., 'bbox': tensor([...])}

            # --- 5. Test Batch Preparation ---
            print("\n--- 5. Testing prepare_batch_inputs (moving to device) ---")
            model_inputs = prepare_batch_inputs(batched_inputs, device)
            
            print(f"  Model Inputs Keys: {model_inputs.keys()}")
            print(f"  src_sketch shape on device: {model_inputs['src_sketch'].shape}")
            print(f"  src_sketch_mask shape on device: {model_inputs['src_sketch_mask'].shape}")
            print(f"  src_video shape on device: {model_inputs['src_video'].shape}")
            print(f"  src_video_mask shape on device: {model_inputs['src_video_mask'].shape}")
            
            print("\n✅ --- Dataset Test Passed --- ✅")
        except Exception as e:
            print(f"\n❌ --- Dataset Test Failed --- ❌")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()