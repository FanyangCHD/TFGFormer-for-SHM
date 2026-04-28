import torch
import torch.nn as nn
import torch.nn.functional as F

class JointTimeFreqLoss(nn.Module):

    def __init__(self, spatial_weight=1.0, freq_weight=1):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.freq_weight = freq_weight
    
    def forward(self, pred, target):

        spatial_loss = F.mse_loss(pred, target)
        
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        real_loss = F.mse_loss(pred_fft.real, target_fft.real)
        imag_loss = F.mse_loss(pred_fft.imag, target_fft.imag)
        freq_loss = (real_loss + imag_loss) / 2
        
        return self.spatial_weight * spatial_loss + self.freq_weight * freq_loss
