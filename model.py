import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Optional
from torch import Tensor
from torchvision.models import resnet34, ResNet34_Weights

# ---
# 1. lib/modeling/position_encoding.py
# ---

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images. (To 1D sequences)
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        """
        Args:
            x: torch.tensor, (batch_size, L, d)
            mask: torch.tensor, (batch_size, L), with 1 as valid
        """
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # (bsz, L)
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t  # (bsz, L, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # (bsz, L, num_pos_feats)
        return pos_x

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat, mask=None):
        """
        Args:
            input_feat: (N, L, D)
        """
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def build_position_encoding(args):
    N_steps = args.hidden_dim
    if args.sketch_position_embedding == 'trainable':
        sketch_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=args.num_input_sketches, # This is 1
            hidden_size=args.hidden_dim,
            dropout=args.input_dropout
        )
    elif args.sketch_position_embedding == 'sine':
        sketch_pos_embed = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.sketch_position_embedding}")

    if args.video_position_embedding == 'trainable':
        video_pos_embed = TrainablePositionalEncoding(
            max_video_position_embeddings=args.num_input_frames, # This is 32 * 7 * 7
            hidden_size=args.hidden_dim,
            dropout=args.input_dropout
        )
    elif args.video_position_embedding == 'sine':
        video_pos_embed = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {args.video_position_embedding}")

    return sketch_pos_embed, video_pos_embed

# ---
# 2. lib/modeling/cross_modal_transformer.py
# ---

class CrossModalMLP(nn.Module): # Renamed from MLP to avoid conflict
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, activation="gelu"):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = _get_activation_fn(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CrossModalTransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, activation="relu"):
        super().__init__()
        self.sketch_video_cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.content_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp1 = CrossModalMLP(in_features=d_model, hidden_features=dim_feedforward, activation=activation)
        self.norm3 = nn.LayerNorm(d_model)

        self.token_self_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm4 = nn.LayerNorm(d_model)
        self.content_token_cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.norm5 = nn.LayerNorm(d_model)
        self.mlp2 = CrossModalMLP(in_features=d_model, hidden_features=dim_feedforward, activation=activation)
        self.norm6 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src_vid, src_skch, out, vid_mask = None, skch_mask = None,
              vid_pos = None, skch_pos=None, query_pos = None):
        '''
        Args:
            src_vid: [L_vid, bs, d]
            src_skch: [L_skch, bs, d] (L_skch is 1 in our case)
            out: [num_queries, bs, d]
            vid_mask: [bs, L_vid] (Boolean mask, True for PADDED tokens)
            skch_mask: [bs, L_skch] (Boolean mask, True for PADDED tokens)
            vid_pos: [L_vid, bs, d]
            skch_pos: [L_skch, bs, d]
            query_pos: [num_queries, bs, d]
        '''
        q = src_skch
        k = v = self.with_pos_embed(src_vid, vid_pos)
        
        # ---
        # ** FIX 1: Add key_padding_mask here **
        # ---
        _, att1 = self.sketch_video_cross_attn(q, k, v, key_padding_mask=vid_mask)
        
        mem = att1.permute(2, 0, 1) * src_vid
        mem = src_vid + mem
        mem = self.norm1(mem)

        q = k = self.with_pos_embed(mem, vid_pos)
        v = mem
        
        # ---
        # ** FIX 2: Add key_padding_mask here **
        # ---
        mem, att2 = self.content_self_attn(q, k, v, key_padding_mask=vid_mask)
        
        mem = mem + v
        mem = self.norm2(mem)
        mem = mem + self.mlp1(mem)
        mem = self.norm3(mem)

        q = k = self.with_pos_embed(out, query_pos)
        v = out
        # This layer (token self-attn) doesn't need a mask, as queries are not padded
        out, att3 = self.token_self_attn(q, k, v)
        out = out + v
        out = self.norm4(out)

        q = self.with_pos_embed(out, query_pos)
        k = self.with_pos_embed(mem, vid_pos)
        v = mem
        # This layer already had the mask, which is correct
        out2, att4 = self.content_token_cross_attn(q, k, v, key_padding_mask=vid_mask)
        out = out + out2
        out = self.norm5(out)
        out = out + self.mlp2(out)
        out = self.norm6(out)

        return mem, out, att1, att2, att3, att4

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, activation="gelu"):
        super().__init__()
        layer = CrossModalTransformerLayer(d_model, nhead, dim_feedforward, activation)
        self.layers = _get_clones(layer, num_layers)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_skch, vid_mask, skch_mask, vid_pos, skch_pos, query_embed):
        outputs = []
        att1_weights = []  # sketch-video cross attention
        att2_weights = []  # content self attention
        att3_weights = []  # token self attention
        att4_weights = []  # content-token cross attention

        bs, l_vid, d = src_vid.shape
        _, l_skch, _ = src_skch.shape
        
        src_vid = src_vid.transpose(0, 1)    # (L_vid, bs, d)
        src_skch = src_skch.transpose(0, 1)  # (L_skch, bs, d)
        vid_pos = vid_pos.transpose(0, 1)    # (L_vid, bs, d)
        skch_pos = skch_pos.transpose(0, 1)  # (L_skch, bs, d)
        
        if len(query_embed.shape) != 3:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, bs, d)
        
        mem = src_vid
        out = torch.zeros_like(query_embed)  # (#queries, bs, d)

        for layer in self.layers:
            mem, out, att1, att2, att3, att4 = layer(
                mem, src_skch, out,
                vid_mask=vid_mask, skch_mask=skch_mask,
                vid_pos=vid_pos, skch_pos=skch_pos,
                query_pos=query_embed
            )
            outputs.append(out)
            att1_weights.append(att1)
            att2_weights.append(att2)
            att3_weights.append(att3)
            att4_weights.append(att4)

        outputs = torch.stack(outputs).transpose(1, 2)  # (#layers, bs, #queries, d)
        att1_weights = torch.stack(att1_weights)  # (#layers, bs, L_skch, L_vid)
        att2_weights = torch.stack(att2_weights)  # (#layers, bs, L_vid, L_vid)
        att3_weights = torch.stack(att3_weights)  # (#layers, bs, #queries, #queries)
        att4_weights = torch.stack(att4_weights)  # (#layers, bs, #queries, L_vid)

        return outputs, att1_weights, att2_weights, att3_weights, att4_weights

def build_cross_modal_transformer(args):
    return CrossModalTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
    )

# ---
# 3. lib/modeling/svanet.py
# ---

class SVANetMLP(nn.Module): # Renamed from MLP
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SVANetLinearLayer(nn.Module): # Renamed from LinearLayer
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(SVANetLinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)

class SVANet(nn.Module):
    """ End-to-End Sketch-based Video Object Localization with Transformer """
    def __init__(self, transformer, sketch_position_embed, video_position_embed,
                 input_vid_dim, input_skch_dim, num_queries, input_dropout=0.1,
                 aux_loss=True, use_sketch_pos=True, n_input_proj=2, num_classes=2, vis_mode=None):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.transformer = transformer
        self.sketch_position_embed = sketch_position_embed
        self.video_position_embed = video_position_embed
        hidden_dim = transformer.d_model
        self.bbox_embed = SVANetMLP(hidden_dim, hidden_dim, 4, 3) # Use renamed MLP
        self.use_sketch_pos = use_sketch_pos
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.n_input_proj = n_input_proj
        self.class_head = nn.Linear(hidden_dim, num_classes)

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_video_proj = nn.Sequential(*[
            SVANetLinearLayer(input_vid_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            SVANetLinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            SVANetLinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_sketch_proj = nn.Sequential(*[
            SVANetLinearLayer(input_skch_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            SVANetLinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            SVANetLinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.vis_mode = vis_mode
        self.aux_loss = aux_loss

    def forward(self, src_sketch, src_sketch_mask, src_video, src_video_mask):
        """
        src_sketch: [batch_size, L_sketch, D_skch] (L_sketch=1)
        src_sketch_mask: [batch_size, L_sketch] (L_sketch=1, 1 is valid)
        src_video: [batch_size, L_video, D_vid] (L_video=T*H*W)
        src_video_mask: [batch_size, L_video] (L_video=T*H*W, 1 is valid)
        """
        # ---
        # ** THE FIX IS HERE **
        # ---
        
        # 1. Project features
        src_video = self.input_video_proj(src_video)  # (bs, L_video, d)
        src_sketch = self.input_sketch_proj(src_sketch)  # (bs, L_sketch, d)

        # 2. Create positional embeddings
        # PositionEmbeddingSine expects a mask where 1 is VALID.
        # Your src_video_mask is already in this format.
        pos_video = self.video_position_embed(src_video, src_video_mask) # (bs, L_video, d)
        pos_sketch = self.sketch_position_embed(src_sketch, src_sketch_mask)  # (bs, L_sketch, d)

        # 3. Create masks for the Transformer
        # MultiheadAttention expects a mask where True is PADDED/INVALID.
        # This is the INVERSE of your dataloader mask.
        mask_video = ~src_video_mask.bool()  # (bs, L_video)
        mask_sketch = ~src_sketch_mask.bool()  # (bs, L_sketch)

        # --- END FIX ---
        
        hs, att1, att2, att3, att4 = self.transformer(
            src_video, src_sketch, mask_video, mask_sketch, pos_video, pos_sketch, self.query_embed.weight
        )

        outputs_class = self.class_embed(hs)  # (#layers, bs, #queries, 2)
        outputs_coord = self.bbox_embed(hs).sigmoid()  # (#layers, bs, #queries, 4)
        
        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        
        return out

def build_svanet(args):
    transformer = build_cross_modal_transformer(args)
    sketch_position_embed, video_position_embed = build_position_encoding(args)

    return SVANet(
        transformer,
        sketch_position_embed,
        video_position_embed,
        input_vid_dim=args.input_vid_dim,
        input_skch_dim=args.input_skch_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        use_sketch_pos=args.use_sketch_pos,
        n_input_proj=args.n_input_proj,
        vis_mode=args.vis_mode,
    )

# ---
# 4. lib/modeling/backbone.py (Simplified)
# ---

class ResNetBackbone(nn.Module):
    def __init__(self, video_backbone, sketch_backbone):
        super(ResNetBackbone, self).__init__()
        self.video_backbone = video_backbone
        self.sketch_backbone = sketch_backbone

    def forward(self, sketch_batch, video_batch):
        '''
        sketch_batch: [N, 1, C, H, W]
        video_batch: [N, T, C, H, W]
        '''
        sketch_batch = sketch_batch.squeeze(1)  # (N, C, H, W)
        # ResNet backbone for sketch (image)
        # Input: (N, C, H, W), Output of [:-1] is (N, 512, 1, 1)
        src_sketch = self.sketch_backbone(sketch_batch).squeeze()  # (N, 512)
        if src_sketch.dim() == 1: # Handle batch size 1
             src_sketch = src_sketch.unsqueeze(0)
        src_sketch = src_sketch.unsqueeze(1)  # (N, 1, 512)

        N, T, C0, H0, W0 = video_batch.shape
        video_batch = video_batch.flatten(0, 1)  # (N*T, C, H, W)
        # ResNet backbone for video frames
        # Input: (N*T, C, H, W), Output of [:-2] is (N*T, 512, 7, 7)
        src_video = self.video_backbone(video_batch)  
        
        N_T, C, H, W = src_video.shape
        src_video = src_video.reshape(N, T, C, H, W) # (N, T, 512, 7, 7)
        src_video = src_video.permute(0, 2, 1, 3, 4)  # (N, 512, T, 7, 7)
        src_video = src_video.flatten(2, -1)  # (N, 512, T*H*W)
        src_video = src_video.transpose(1, 2)  # (N, T*H*W, 512)

        return src_sketch, src_video

def build_backbone(args):
    # We hardcode ResNet-34 as it's a good default
    video_backbone = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-2])
    sketch_backbone = nn.Sequential(*list(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).children())[:-1])

    # Set dims for the args object so build_svanet can use them
    args.input_vid_dim = 512
    args.input_skch_dim = 512
    
    backbone = ResNetBackbone(
        video_backbone,
        sketch_backbone
    )
    return backbone

# ---
# 5. lib/modeling/model.py
# ---

class SketchLocalizationModel(nn.Module):
    def __init__(self, backbone, head):
        super(SketchLocalizationModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, src_sketch, src_video, src_sketch_mask=None, src_video_mask=None):
        N, T, C0, H0, W0 = src_video.shape
        
        # 1. Backbone
        # src_sketch: (N, 1, 512)
        # src_video: (N, T*H*W, 512) -> (N, 32*7*7, 512) -> (N, 1568, 512)
        src_sketch, src_video = \
            self.backbone(src_sketch, src_video)  

        # 2. Adjust Masks
        # src_sketch_mask is (N, 1), which is correct.
        
        # src_video_mask is (N, T). We need (N, T*H*W)
        # src_video.shape[1] is T*H*W (1568)
        # T is 32
        # H*W is src_video.shape[1] // T = 1568 // 32 = 49
        feature_map_size = src_video.shape[1] // T
        src_video_mask = src_video_mask.repeat_interleave(feature_map_size, dim=1)

        # 3. Head
        outputs = self.head(
            src_sketch, src_sketch_mask,
            src_video, src_video_mask
        )
        return outputs

def build_model(args):
    backbone = build_backbone(args)
    if args.sketch_head == 'svanet':
        head = build_svanet(args)
    else:
        raise NotImplementedError

    model = SketchLocalizationModel(
        backbone,
        head
    )
    return model

# ---
# 6. TEST SCRIPT
# ---

# if __name__ == "__main__":
    
#     print("--- 1. Initializing Model ---")

#     # 1. Create a mock 'args' object to hold model config
#     class MockArgs:
#         # --- Model Config ---
#         backbone = 'resnet'
#         sketch_head = 'svanet'
#         hidden_dim = 256
#         nheads = 8
#         num_layers = 4        # From configs.py default
#         dim_feedforward = 1024 # From configs.py default
        
#         # --- Query Config ---
#         # 32 frames, 10 queries per frame = 320 total queries
#         num_queries = 320     
#         num_input_frames = 32 * 7 * 7 # T * H * W
#         num_input_sketches = 1
        
#         # --- Embeddings & Projections ---
#         input_dropout = 0.4
#         n_input_proj = 2
#         dropout = 0.1
#         pre_norm = False
#         use_sketch_pos = True
#         sketch_position_embedding = 'sine'
#         video_position_embedding = 'sine'
        
#         # --- Loss & Other ---
#         aux_loss = True
#         vis_mode = None
        
#         # --- Dims (will be set by build_backbone) ---
#         input_vid_dim = None
#         input_skch_dim = None

#     args = MockArgs()
    
#     # 2. Build the model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = build_model(args)
#     model.to(device)
#     model.train() # Set to train mode

#     print(f"Model built successfully. Running on {device}.")
#     print(f"  Total Queries: {args.num_queries}")
#     print(f"  Hidden Dim: {args.hidden_dim}")
#     print(f"  Backbone Dims: Vid={args.input_vid_dim}, Sketch={args.input_skch_dim}")

#     # 3. Create dummy inputs
#     print("\n--- 2. Creating Dummy Input ---")
#     BATCH_SIZE = 2
#     NUM_FRAMES = 32
    
#     # (N, T, C, H, W)
#     dummy_video = torch.rand(BATCH_SIZE, NUM_FRAMES, 3, 224, 224).to(device)
#     # (N, T) - Mask is 1 for valid, 0 for padding. All valid here.
#     dummy_video_mask = torch.ones(BATCH_SIZE, NUM_FRAMES).to(device) 
    
#     # (N, 1, C, H, W)
#     dummy_image = torch.rand(BATCH_SIZE, 1, 3, 224, 224).to(device)
#     # (N, 1) - All valid
#     dummy_image_mask = torch.ones(BATCH_SIZE, 1).to(device)

#     print(f"  Video Shape: {dummy_video.shape}")
#     print(f"  Image Shape: {dummy_image.shape}")

#     # 4. Run forward pass
#     print("\n--- 3. Running Forward Pass ---")
#     with torch.no_grad(): # We don't need gradients for this test
#         outputs = model(dummy_image, dummy_video, dummy_image_mask, dummy_video_mask)

#     # 5. Print output shapes
#     print("\n--- 4. Inspecting Outputs ---")
#     print(f"  Output keys: {outputs.keys()}")
    
#     # Check main output
#     print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
#     print(f"  pred_boxes shape: {outputs['pred_boxes'].shape}")
    
#     assert outputs['pred_logits'].shape == (BATCH_SIZE, args.num_queries, 2)
#     assert outputs['pred_boxes'].shape == (BATCH_SIZE, args.num_queries, 4)

#     # Check auxiliary outputs
#     print(f"  aux_outputs length: {len(outputs['aux_outputs'])}")
#     assert len(outputs['aux_outputs']) == args.num_layers - 1
    
#     print(f"  aux_outputs[0] logits shape: {outputs['aux_outputs'][0]['pred_logits'].shape}")
#     print(f"  aux_outputs[0] boxes shape: {outputs['aux_outputs'][0]['pred_boxes'].shape}")
#     assert outputs['aux_outputs'][0]['pred_logits'].shape == (BATCH_SIZE, args.num_queries, 2)
#     assert outputs['aux_outputs'][0]['pred_boxes'].shape == (BATCH_SIZE, args.num_queries, 4)
    
#     print("\n✅ --- Model Test Passed --- ✅")