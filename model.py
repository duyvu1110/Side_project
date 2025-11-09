import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class ZaloTrackerNet(nn.Module):
    """
    A Query-based Video Object Detection model.
    
    This model takes a single query image and a video clip (T frames)
    and predicts bounding boxes for the query object in every frame.
    
    It uses a shared ResNet-34 backbone to extract features, "conditions"
    the video features with the query, and then uses detection heads
    to predict object presence and location on a 7x7 grid.
    """
    def __init__(self, feature_dim=256, dropout=0.1):
        super(ZaloTrackerNet, self).__init__()
        
        # --- 1. Load Pre-trained ResNet-34 Backbone ---
        
        # Get a pre-trained ResNet-34
        weights = ResNet34_Weights.DEFAULT
        resnet = resnet34(weights=weights)
        
        # We will share the weights for query and video processing.
        # We don't need the final avgpool and fc layers.
        # We'll take the output of 'layer4' (the last conv block).
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4 
        )
        
        # The output of ResNet-34's layer4 has 512 channels.
        # Input: (B, 3, 224, 224)
        # Output: (B, 512, 7, 7)
        
        # --- 2. Query Condensing Layer ---
        
        # This compresses the query's 7x7 feature map into a single
        # feature vector using global average pooling.
        self.query_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 3. Feature Fusion / Conditioning ---
        
        # This 1x1 conv projects the 512-dim backbone features
        # down to a smaller `feature_dim` for the detection heads.
        self.input_proj = nn.Conv2d(512, feature_dim, kernel_size=1)
        
        # --- 4. Prediction Heads ---
        
        # These heads will be applied to the "fused" features.
        
        # Classification Head: Predicts "object-ness" score
        # For each of the 7x7=49 locations, is the object here?
        self.cls_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1) # 1 output channel for score
        )
        
        # Regression Head: Predicts bounding box [cx, cy, w, h]
        # For each of the 7x7=49 locations, what is the box?
        self.reg_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 4, kernel_size=1), # 4 output channels for box
            nn.Sigmoid() # Force box coords to be between 0 and 1
        )

    def forward(self, input_query_image, input_video):
        """
        Forward pass of the model.
        
        Args:
            input_query_image (Tensor): Shape [B, 1, 3, 224, 224]
            input_video (Tensor): Shape [B, T, 3, 224, 224] (e.g., T=32)
            
        Returns:
            dict: {
                'pred_logits': [B, T, 49, 1],  (Raw scores for 49 locations)
                'pred_boxes': [B, T, 49, 4]   (Normalized [cx, cy, w, h] boxes)
            }
        """
        
        # --- 1. Process the Query Image ---
        
        # Shape: [B, 1, 3, 224, 224] -> [B, 3, 224, 224]
        query_img = input_query_image.squeeze(1) 
        
        # Get query features: [B, 512, 7, 7]
        query_feat_map = self.backbone(query_img) 
        
        # "Condense" query to a single vector: [B, 512, 1, 1]
        # This vector represents the "visual signature" of the query.
        query_vec = self.query_pool(query_feat_map) 

        # --- 2. Process the Video Frames ---
        
        # Get video dimensions
        B, T, C, H, W = input_video.shape # B=Batch, T=Time
        
        # Flatten batch and time dims to process all frames at once
        # Shape: [B, T, 3, 224, 224] -> [B*T, 3, 224, 224]
        video_flat = input_video.view(B * T, C, H, W)
        
        # Get video features: [B*T, 512, 7, 7]
        video_feat_map = self.backbone(video_flat)
        
        # --- 3. Fuse Query and Video Features ---
        
        # "Condition" the video features with the query vector.
        # We use element-wise multiplication.
        
        # Expand query_vec to match video_feat_map dimensions for broadcasting
        # query_vec: [B, 512, 1, 1]
        #   -> unsqueeze(1): [B, 1, 512, 1, 1]
        #   -> repeat:       [B, T, 512, 1, 1]
        #   -> view:         [B*T, 512, 1, 1]
        query_vec_broadcast = query_vec.unsqueeze(1).repeat(1, T, 1, 1, 1).view(B * T, 512, 1, 1)
        
        # Fused features = Video features "filtered" by the Query
        # Shape: [B*T, 512, 7, 7]
        fused_features = video_feat_map * query_vec_broadcast
        
        # --- 4. Project Features ---
        
        # Project down to a smaller dimension
        # Shape: [B*T, 256, 7, 7]
        fused_features_proj = self.input_proj(fused_features)

        # --- 5. Get Predictions from Heads ---
        
        # Classification logits
        # Shape: [B*T, 1, 7, 7]
        pred_logits_map = self.cls_head(fused_features_proj)
        
        # Bounding box regression
        # Shape: [B*T, 4, 7, 7]
        pred_boxes_map = self.reg_head(fused_features_proj)
        
        # --- 6. Reshape for Output ---
        
        # We need to separate Batch (B) and Time (T) again
        
        # Logits: [B*T, 1, 7, 7]
        #   -> view:   [B, T, 1, 7, 7]
        #   -> flatten: [B, T, 1, 49]
        #   -> permute: [B, T, 49, 1] (Batch, Time, Location, Score)
        pred_logits = pred_logits_map.view(B, T, 1, 7, 7).flatten(3).permute(0, 1, 3, 2)
        
        # Boxes: [B*T, 4, 7, 7]
        #   -> view:   [B, T, 4, 7, 7]
        #   -> flatten: [B, T, 4, 49]
        #   -> permute: [B, T, 49, 4] (Batch, Time, Location, Box)
        pred_boxes = pred_boxes_map.view(B, T, 4, 7, 7).flatten(3).permute(0, 1, 3, 2)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }