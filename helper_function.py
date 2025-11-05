import cv2
import numpy as np
from PIL import Image

def load_video_frames_at_indices(video_path, sampled_idxs):
    """
    Efficiently loads specific video frames from a video file.
    
    Args:
        video_path (str): Path to the .mp4 file.
        sampled_idxs (list[int]): A list of frame indices to load.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    frames = []
    for frame_idx in sampled_idxs:
        # Set the video to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # If reading fails, try reading the previous frame (robustness)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret:
                # If it still fails, use a black frame
                print(f"Warning: Error reading frame {frame_idx} from video: {video_path}. Using black frame.")
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # Convert from BGR (OpenCV) to RGB (PIL/Torch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames