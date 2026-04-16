import random, torch, os
import numpy as np
import torch.nn as nn
from typing import Optional
import numpy as np
import pickle
import cv2
from scipy import ndimage
from scipy.interpolate import griddata


class TrivialUpdater(nn.Module):
    """
    Trivial update that directly returns logits-weighted embeddings.
    
    Efficiently handles tensors of arbitrary shape (..., vocab_size), preserving
    all leading dimensions while computing weighted embeddings.
    """

    def __init__(self, use_hidden_states: bool = False, topk: Optional[int] = None):
        super().__init__()
        self.use_hidden_states = use_hidden_states
        self.topk = topk # 100

    def forward(
        self,
        logits: torch.Tensor,
        prev_inputs: torch.Tensor,
        embedding_weight: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Direct matrix multiplication preserves all leading dimensions: (..., vocab_size) @ (vocab_size, embed_dim) -> (..., embed_dim)
        if self.use_hidden_states:
            return hidden_states[...,-1,:] # shape: seq_len, num_layer, embed_dim
        else:
            if self.topk is not None:
                topk_values, topk_indices = torch.topk(logits, k=min(self.topk, logits.size(-1)), dim=-1)
                topk_probs = torch.softmax(topk_values, dim=-1)
                topk_embeddings = embedding_weight[topk_indices]
                return torch.sum(topk_probs.unsqueeze(-1) * topk_embeddings, dim=-2)
            else:
                return torch.softmax(logits, dim=-1) @ embedding_weight


def save_tensor_as_png(tensor, path):

    img = tensor.detach().cpu().numpy()  # [C, H, W]
    
    if img.max() <= 1.0:
        img = img * 255.0
    elif img.min() < 0:
        img = (img + 1) * 127.5  # [-1,1] -> [0,255]
    
    img = img.clip(0, 255).astype('uint8')
    
    if img.shape[0] == 1:
        img = img.squeeze(0)  # [H, W]
    elif img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # RGB [H, W, C]
    
    pil_img = Image.fromarray(img)
    pil_img.save(path, format='PNG')


class TokenSRNet(nn.Module):
    def __init__(self, mid_channels=512):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(1536, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 
                               kernel_size=3, stride=3, 
                               padding=1, groups=mid_channels, 
                               bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, 1536, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(1536, 1536, kernel_size=3, padding=1, 
                      groups=1536, bias=True),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.compress(x)      # [B, 512, 4, 4]
        x = self.upsample(x)      # [B, 512, 10, 10]
        x = self.expand(x)        # [B, 1536, 10, 10]
        x = self.refine(x)        # [B, 1536, 10, 10]
        return x

class TransposeConvSuperRes(nn.Module):
    def __init__(self, in_channels=1536, hidden_channels=768):
        super().__init__()
        
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=2, stride=2, padding=0),
            nn.LayerNorm([hidden_channels, 8, 8]),
            nn.GELU(),
            
            nn.Conv2d(hidden_channels, hidden_channels, 
                     kernel_size=3, padding=1),
            nn.LayerNorm([hidden_channels, 8, 8]),
            nn.GELU(),
            
            nn.ConvTranspose2d(hidden_channels, in_channels, 
                             kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([in_channels, 8, 8]),
            nn.GELU(),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: (4, 4, 1536) or (B, 4, 4, 1536)
        output：(10, 10, 1536) or (B, 10, 10, 1536)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False
        
        x = x.permute(0, 3, 1, 2)  # (B, 1536, 4, 4)
        
        residual = self.residual_conv(x)  # (B, 1536, 4, 4)
        residual = F.interpolate(residual, size=(10, 10), mode='bilinear', align_corners=False)  # (B, 1536, 10, 10)
        
        # (B, 1536, 4, 4) -> (B, 1536, 8, 8) -> (B, 1536, 8, 8)
        x = self.trans_conv(x)
        
        # (B, 1536, 8, 8) -> (B, 1536, 10, 10)
        x = self.adaptive_pool(x)
        
        x = x + residual  # (B, 1536, 10, 10)
        
        # (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, 10, 10, 1536)
        
        if squeeze_out and x.size(0) == 1:
            x = x.squeeze(0)
        
        return x

     

class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
