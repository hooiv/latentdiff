# latentdiff

# Latent Diffusion Style Transfer

This project implements a latent diffusion model for high-resolution artistic style transfer. The model applies the artistic style of one image to the content of another, operating in the latent space for efficiency.

## Project Structure

- `src/` - Source code for model, training, and evaluation
- `data/` - Scripts and instructions for dataset preparation
- `notebooks/` - Jupyter notebooks for experiments and visualizations
- `results/` - Generated images and evaluation outputs
- `requirements.txt` - Python dependencies

## Key Components

- **Latent Diffusion Model:** Efficiently handles high-resolution images in the latent space. See `src/model.py` for architecture details and custom modules.
- **Conditioning Mechanism:** Incorporates style information for effective transfer. Details and implementation in `src/model.py`.
- **Custom Components:** (Optional) Custom noise schedules or samplers for improved quality. Documented in `src/model.py` and `notebooks/`.

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Prepare datasets as described in `data/README.md`
3. Train the model: `python src/train.py`
4. Run style transfer: `python src/infer.py --content <content_img> --style <style_img>`

## Documentation

- Model architecture and custom components are documented in `src/model.py` and `docs/`.
- Visualizations and results are in `notebooks/` and `results/`.

