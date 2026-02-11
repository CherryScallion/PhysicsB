# models/loss.py
import torch
import torch
import torch.nn as nn

class WeightsRegressionLoss(nn.Module):
    def __init__(self, lambda_cos=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.lambda_cos = lambda_cos

    def forward(self, pred, target):
        """
        pred:   [Batch, 64]
        target: [Batch, 64]
        """
        loss_mse = self.mse(pred, target)
        
        cos_sim = self.cosine(pred, target).mean() 
        loss_cos = 1.0 - cos_sim
        
        # combine
        total_loss = loss_mse + self.lambda_cos * loss_cos
        
        return total_loss, {"mse": loss_mse.item(), "cos": loss_cos.item()}