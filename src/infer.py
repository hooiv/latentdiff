import torch
from model import LatentDiffusionModel, ddim_sampler
from PIL import Image
import argparse
import numpy as np

def load_image(path, image_size=512):
    img = Image.open(path).convert('RGB')
    img = img.resize((image_size, image_size))
    img = torch.from_numpy(np.array(img)).float().permute(2,0,1) / 255.0
    return img.unsqueeze(0)

def save_image(tensor, path):
    img = tensor.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = (img * 255).clip(0,255).astype('uint8')
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--ckpt', type=str, default='latentdiff.pth')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LatentDiffusionModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    content = load_image(args.content).to(device)
    style = load_image(args.style).to(device)
    with torch.no_grad():
        out_latent = ddim_sampler(model, content, style, steps=50)
    save_image(out_latent, args.output)
    print(f"Saved stylized image to {args.output}")

if __name__ == "__main__":
    main()
