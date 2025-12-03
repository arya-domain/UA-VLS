import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class SEU_Loss(nn.Module):
    def __init__(self, fourier_weight=0.3, entropy_weight=0.1):
        """
        :param fourier_weight: weight for the Fourier loss component
        :param entropy_weight: weight for the entropy-based uncertainty component
        """
        super(LossFunction, self).__init__()
        self.fourier_weight = fourier_weight
        self.entropy_weight = entropy_weight

    def forward(self, prediction, target):
        """
        :param prediction: Tensor [B, C, H, W], softmax or sigmoid applied
        :param target: Tensor [B, H, W] or [B, C, H, W] (one-hot or label)
        :return: scalar loss
        """
        B, C, H, W = prediction.shape

        # Convert label to one-hot if needed
        if target.dim() == 3:
            target = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
        else:
            target = target.float()

        # -------- Dice Loss --------
        pred_flat = prediction.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice_score.mean()

        # -------- Fourier Loss --------
        pred_fft = torch.fft.fft2(prediction, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        fourier_loss = F.mse_loss(pred_mag, target_mag)

        # -------- Entropy-based Uncertainty Loss --------
        if self.entropy_weight > 0:
            entropy = -torch.sum(prediction * torch.log(prediction + 1e-8), dim=1)  # [B, H, W]
            uncertainty_loss = entropy.mean()
        else:
            uncertainty_loss = 0.0

        # -------- Total Loss --------
        total_loss = (
            dice_loss
            + self.fourier_weight * fourier_loss
            + self.entropy_weight * uncertainty_loss
        )

        return total_loss