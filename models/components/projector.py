import torch
import torch.nn as nn
import torch.nn.functional as F

class BasisProjector(nn.Module):
    """
    [Module 3 & 4: Basis Projection & Semantic Segmentation]
    
    Pipeline:
    1. Linear Combination: Weighted Sum(Basis, z_bold) ->  [B, 1, D, H, W]
    2. Segmentation Head: 1x1 Conv3d -> Logits [B, 3, D, H, W]
    """
    def __init__(self, basis_buffer_name='spatial_basis'):
        super().__init__()
        self.basis_buffer_name = basis_buffer_name
        
        self.classifier = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=1), # up
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=1) 
        )

    def forward(self, weights, model_buffer):
        """
        weights: [Batch, K] 
        model_buffer: self of main model，get buffer
        """
        # 1. load ICA basis
        # Shape: [K, D, H, W]
        if not hasattr(model_buffer, self.basis_buffer_name):
            raise RuntimeError(f"Buffer {self.basis_buffer_name} not found in model!")
            
        basis_maps = getattr(model_buffer, self.basis_buffer_name)
        
        # Vector to Volume
        # w[b, k] * map[k, d, h, w] -> vol[b, d, h, w]
        rough_volume = torch.einsum('bk, kdhw -> bdhw', weights, basis_maps)
        # [B, D, H, W] -> [B, 1, D, H, W]
        x = rough_volume.unsqueeze(1)
        # Logits: [B, 3, D, H, W]
        logits = self.classifier(x)
        
        return logits