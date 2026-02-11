import torch
import torch.nn as nn
import numpy as np

class HRFBlock(nn.Module):
    """
    [Module 2: The Physics Aligner]
    """
    def __init__(self, n_components, learnable=False):
        super().__init__()
        self.learnable = learnable
        
        if self.learnable:
            # [1, 1, 3] convo kernel
            # temporal smoothing
            self.smoothing = nn.Conv1d(
                in_channels=n_components, 
                out_channels=n_components,
                kernel_size=3,
                padding=1,
                groups=n_components # Depthwise conv
            )
            # [0.2, 0.6, 0.2]
            nn.init.constant_(self.smoothing.weight, 0.2)
            self.smoothing.weight.data[:, 0, 1] = 0.6
        
    def forward(self, z_neural):
        """
        Input: [Batch, K] 
            or [Batch, Time, K] 
        """
        if not self.learnable:
            return z_neural
            
        if z_neural.dim() == 2:
            return z_neural #
            
        # [B, T, K] -> [B, K, T]
        x = z_neural.permute(0, 2, 1)
        x = self.smoothing(x)
        # Back to [B, T, K]
        return x.permute(0, 2, 1)

def spm_hrf_basis(tr, duration=32.0):
    """
    (Optional Tool)
    SPM HRF kernel(do I need this in the future work?)
    """
    dt = tr
    t = np.arange(0, duration, dt)
    
    # SPM default parameters
    p = [6, 16, 1, 1, 6, 0] 
    
    # Double Gamma Function
    # h = (t/d1)^a1 * exp(-(t-d1)/b1) - c * (t/d2)^a2 * exp(-(t-d2)/b2)
    from scipy.stats import gamma
    
    hrf = gamma.pdf(t, p[0]/p[2], scale=dt*p[2]) - \
          p[5] * gamma.pdf(t, p[1]/p[3], scale=dt*p[3])
          
    hrf = hrf / np.sum(hrf)
    return torch.tensor(hrf).float()