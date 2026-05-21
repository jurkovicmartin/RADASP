import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalDiceLoss(nn.Module):
    """
    Hybrid loss function combining Focal Loss and Dice Loss.
    """
    def __init__(self, weight: float =0.5, alpha: float =0.75, gamma: float =2, smooth: float =1e-6):
        """
        Args:
            weight (float): Weighting factor between Focal and Dice loss. 
                            Higher values prioritize Focal loss. Defaults to 0.5.
            alpha (float): Focal loss alpha parameter for class balancing. Defaults to 0.75.
            gamma (float): Focal loss gamma parameter for focusing on hard examples. Defaults to 2.
            smooth (float): Smoothing factor to avoid division by zero in Dice calculation. Defaults to 1e-6.
        """
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        input = F.sigmoid(input)

        target = target.flatten()
        input = input.flatten()

        intersection = (input * target).sum()                            
        dice = 1 - (2.*intersection + self.smooth) / (input.sum() + target.sum() + self.smooth)

        bce = F.binary_cross_entropy(input, target.float(), reduction="none")
        bce_exp = torch.exp(-bce)
        alpha = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal = alpha * (1 - bce_exp)**self.gamma * bce
        focal = torch.mean(focal)
        
        return (self.weight * focal) + ((1 - self.weight) * dice)
