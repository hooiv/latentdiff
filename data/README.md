# Data preparation scripts and instructions

# Data preparation instructions for latent diffusion style transfer

## Content and Style Datasets

- Place your content images in `data/content/`
- Place your style images in `data/style/`
- Images should be in `.jpg` or `.png` format and have matching filenames for pairing

## Example Directory Structure

```
data/
  content/
    img1.jpg
    img2.jpg
    ...
  style/
    img1.jpg
    img2.jpg
    ...
```

## Notes
- Images will be resized to 512x512 by default.
- You may use any dataset of artworks and photos (e.g., WikiArt, COCO, etc.).
- For best results, ensure content and style images are visually diverse.
