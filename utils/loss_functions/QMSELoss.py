import torch
import torch.nn as nn


class QMSELoss(nn.Module):
    def __init__(self, balance=None):
        self.balance = balance

    def __call__(self, prediction_batch, target_batch, bins):
        loss_quantized = 0
        bins = bins.int()
        bins_unique = torch.unique(bins)
        for b in bins_unique:
            mask_b = (bins == b)
            count_b = mask_b.sum()
            loss_b = self.mse_loss(prediction_batch[mask_b], target_batch[mask_b])

            # Bin balancing
            if self.balance == "count":
                weight = 1.0 / count_b
            elif self.balance == "sqrt":
                weight = 1.0 / torch.sqrt(count_b.float())
            else:
                weight = 1.0
            loss_quantized += weight * loss_b

        return loss_quantized


