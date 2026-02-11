# models/components/temporal.py
import torch
import torch.nn as nn

class EEGEncoder(nn.Module):
    """
    [Upgrade] 2D CNN Encoder for EEG Spectrograms
    Input: [Batch, Channels, Freq_Bins, Time_Steps]
           [B, 20, 64, 249] - channels, frequency bins, time steps
    Output: [B, n_components] - ICA weight predictions
    """
    def __init__(self, in_channels, n_components, hidden_dim=64):
        super().__init__()
        
        self.conv2d_net = nn.Sequential(
            # [B, 20, 64, 249]
            nn.Conv2d(in_channels, 32, kernel_size=(3, 7), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.AdaptiveAvgPool2d((1, 1)) # -> [B, 128, 1, 1]
        )
        
        self.proj_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_components) # -> Output Weights
        )

    def forward(self, x):
        """
        x: [Batch, Channels, Freq_Bins, Time_Steps]
        Returns: [Batch, n_components]
        """
        # x should be [B, C, F, T]
        # For 2D CNN, we treat F (freq bins) as height and T (time) as width
        
        feat = self.conv2d_net(x)
        out = self.proj_head(feat)
        return out