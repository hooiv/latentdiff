import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Encoder(nn.Module):
    """
    Encoder network to map images to latent space.
    Uses a pretrained VGG19 as a feature extractor.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.enc_layers = nn.Sequential(*list(vgg.children())[:35])
        for param in self.enc_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.enc_layers(x)

class StyleConditioning(nn.Module):
    """
    Conditioning mechanism to inject style information into the latent space.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, style_latent):
        return self.fc(style_latent)

class DiffusionUNet(nn.Module):
    """
    U-Net backbone for denoising in latent space.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.down1 = nn.Conv2d(latent_dim, 128, 3, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, latent_dim, 3, padding=1)

    def forward(self, x):
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        u1 = F.relu(self.up1(d2))
        u2 = self.up2(u1 + d1)
        return u2

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model for high-resolution style transfer.
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.style_condition = StyleConditioning(latent_dim)
        self.denoiser = DiffusionUNet(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, content_img, style_img, t, noise):
        # Encode images to latent space
        content_latent = self.encoder(content_img)
        style_latent = self.encoder(style_img)
        # Global average pool to vector
        style_vec = F.adaptive_avg_pool2d(style_latent, (1,1)).view(style_latent.size(0), -1)
        style_cond = self.style_condition(style_vec)
        # Add noise to content latent
        noisy_latent = content_latent + noise
        # Inject style conditioning (broadcast to spatial dims)
        style_cond_broadcast = style_cond.unsqueeze(-1).unsqueeze(-1).expand_as(noisy_latent)
        conditioned_latent = noisy_latent + style_cond_broadcast
        # Denoise
        out = self.denoiser(conditioned_latent)
        return out

# Custom noise schedule (cosine)
def cosine_noise_schedule(t, s=0.008):
    import math
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2

# Custom sampler (DDIM-like)
def ddim_sampler(model, content_img, style_img, steps=50, eta=0.0):
    device = content_img.device
    latent_shape = model.encoder(content_img).shape
    x = torch.randn(latent_shape, device=device)
    for i in reversed(range(steps)):
        t = torch.full((content_img.size(0),), i / steps, device=device)
        noise = x
        x = model(content_img, style_img, t, noise)
        if eta > 0:
            x = x + eta * torch.randn_like(x)
    return x
