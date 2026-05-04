import torch
import torch.nn as nn
from .registry import register_loss


@register_loss("PSD_Loss")
class PSD_Loss(nn.Module):
    """
    PSD loss for single-channel 2D fields.
    pred, target: (B, H, W) or (B, 1, H, W)
    """
    def __init__(
            self,
            eps=1e-6,
            wavenumber_weight=0.0,
            lambda_psd=0.005,
            y_dim=128,
            x_dim=128,
            use_mse=False,
            apply_expm1=False
        ):
        super().__init__()
        self.eps = eps
        self.wavenumber_weight = wavenumber_weight
        self.lambda_psd = lambda_psd
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.use_mse = use_mse
        self.apply_expm1 = apply_expm1

    def forward(self, pred, target):
        B = pred.shape[0] // (self.y_dim * self.x_dim)
        if B == 0:
            pred = pred.view(1, self.y_dim, self.x_dim)
            target = target.view(1, self.y_dim, self.x_dim)
            B = 1
        else:
            pred = pred.view(B, self.y_dim, self.x_dim)
            target = target.view(B, self.y_dim, self.x_dim)

        if self.apply_expm1:
            pred = torch.expm1(pred)
            target = torch.expm1(target)

        pred = torch.nan_to_num(pred, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)

        psd_losses = []
        for b in range(B):
            # if (target[b] > 0.1).sum() / (target[b] > 0).sum() < 0.05:
            #     continue
            psd_pred, _ = self._compute_psd_radial(pred[b])
            psd_true, _ = self._compute_psd_radial(target[b])

            loss_b = (torch.log(psd_pred + self.eps) -
                    torch.log(psd_true + self.eps))**2
            
            # Optional wavenumber weighting
            if self.wavenumber_weight != 0:
                k = torch.arange(loss_b.shape[-1], device=loss_b.device)
                loss_b = loss_b * (1 + self.wavenumber_weight * k)

            # # Normalize by total power ---
            # norm = torch.sum(psd_true) + self.eps
            # psd_losses.append(loss_b.mean() / norm)
            psd_losses.append(loss_b.mean())
    
        loss_psd = torch.stack(psd_losses).mean()
        loss_mse = ((pred - target)**2).mean()

        return loss_mse + self.lambda_psd * loss_psd

    def _compute_psd_radial(self, data):
        # data: (H, W)
        data = torch.nan_to_num(data.float(), nan=0.0)

        # 1. FFT2 + shift
        F = torch.fft.fft2(data, norm="ortho")
        F = torch.fft.fftshift(F)

        # 2. Power
        power = (F.real**2 + F.imag**2)

        # 3. Radius grid
        H, W = power.shape
        y = torch.arange(H, device=data.device)
        x = torch.arange(W, device=data.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        center_y = (H - 1) / 2.0
        center_x = (W - 1) / 2.0

        r = torch.sqrt((yy - center_y)**2 + (xx - center_x)**2)
        r = r.to(torch.long)  # indices

        r_flat = r.view(-1)
        p_flat = power.view(-1)

        K = r_flat.max().item() + 1  # number of radial bins

        # 4. Differentiable radial sum and counts
        radial_sum = torch.zeros(K, device=data.device)
        radial_sum.scatter_add_(0, r_flat, p_flat)

        counts = torch.zeros(K, device=data.device)
        counts.scatter_add_(0, r_flat, torch.ones_like(p_flat))

        radial_psd = radial_sum / counts.clamp_min(1)

        wavenumbers = torch.arange(K, device=data.device)
        return radial_psd, wavenumbers
