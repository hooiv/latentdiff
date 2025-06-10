# Training script for latent diffusion style transfer

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import LatentDiffusionModel, ddim_sampler
from PIL import Image
import os

def load_image(path, image_size=512):
    img = Image.open(path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tfm(img).unsqueeze(0)

def train_loop(content_dir, style_dir, epochs=1, batch_size=1, lr=1e-4, device='cuda'):
    model = LatentDiffusionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    content_imgs = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir)])
    style_imgs = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir)])
    for epoch in range(epochs):
        for c_path, s_path in zip(content_imgs, style_imgs):
            content = load_image(c_path).to(device)
            style = load_image(s_path).to(device)
            t = torch.rand(1, device=device)
            noise = torch.randn_like(model.encoder(content))
            out = model(content, style, t, noise)
            loss = ((out - model.encoder(content)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), 'latentdiff.pth')

def main():
    train_loop('data/content', 'data/style')

if __name__ == "__main__":
    main()
