import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(1000, 64)  # Time embedding for diffusion

        # Downsampling layers (Encoder)
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # Downsampling
            nn.ReLU()
        )

        # Bottleneck
        self.middle = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Upsampling layers (Decoder)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # Upsample to match skip connection
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1)  # Ensure channel match for skip connection

        self.up2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # Reduce channels
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Final output layer
        )

    def forward(self, x, t):
        t = t.to(torch.int32)
        t_emb = self.time_embed(t)

        # Downsampling
        h1 = self.down1(x)  # Shape: (B, 64, H, W)
        h2 = self.down2(h1)  # Shape: (B, 128, H/2, W/2)

        # Bottleneck
        h = self.middle(h2)

        # Upsampling
        h = self.up1(h)  # (B, 128, H, W)

        # Fix skip connection size mismatch (if needed)
        if h.shape[2:]!= h1.shape[2:]:
            h = F.interpolate(h, size=h1.shape[2:], mode='bilinear', align_corners=False)

        h = self.conv1x1(h)  # Reduce channels from 128 to 64
        h = h + h1  # Correct skip connection

        # Final upsampling & output
        h = self.up2(h)  # Output shape: (B, 3, H, W)
        return h


# Example usage
model = UNet().cuda()
print(model)


class DDPM(nn.Module):
    def __init__(self, model, T=1000):
        super().__init__()
        self.model = model
        self.T = T
        self.beta = torch.linspace(1e-4, 0.02, T).cuda()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), dtype=torch.int32, device=x0.device)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1) * x0 + \
             torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1) * eps
        pred_eps = self.model(xt, t)
        return nn.MSELoss()(pred_eps, eps)
    
    @torch.no_grad()
    def sample(self, n_samples):
        x = torch.randn(n_samples, 3, 64, 64).cuda()
        for t in reversed(range(self.T)):
            t_tensor = torch.full((n_samples,), t, device="cuda")
            pred_eps = self.model(x, t_tensor)
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]
            alpha_bar_t = self.alpha_bar[t]
            z = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * pred_eps) + torch.sqrt(beta_t) * z
        return x