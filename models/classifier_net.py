# models/classifier_net.py
import torch
import torch.nn as nn
from models.components.temporal import EEGEncoder
from models.components.physics import HRFBlock

class PhysicsB(nn.Module):
    """
    PhysicsB: Physics-Informed Regression Neural Network
    
    Architecture:
    1. EEGEncoder: [B, C, F, T] -> [B, K] (Neural Activity)
    2. HRFBlock: [B, K] -> [B, K] (Hemodynamic Response Filtering)
    3. Output: [B, K] predicted ICA weights
    """
    def __init__(self, 
                 n_ica_components=64,
                 eeg_channels=64,
                 eeg_time_len=300,
                 basis_path=None):  # Kept for backward compatibility, not used
        super().__init__()
        
        # 1. Neural Driver [B, C, F, T] -> [B, K]
        self.driver = EEGEncoder(
            in_channels=eeg_channels,
            n_components=n_ica_components
        )
        
        # 2. Physics Aligner (Hemodynamic Response Filtering)
        self.physics = HRFBlock(n_components=n_ica_components, learnable=False)
        
    def forward(self, eeg_segment):
        """
        Args:
            eeg_segment: [B, C, F, T] (Batch, Channels, Frequency, Time)
        
        Returns:
            [B, K] predicted ICA weights (regression target)
        """
        # Step 1: Encoder -> Latent Weights [B, K]
        z_neural = self.driver(eeg_segment)
        
        # Step 2: Physics Filter (HRF Smoothing) -> [B, K]
        z_bold = self.physics(z_neural)
        
        # Direct weight prediction
        return z_bold