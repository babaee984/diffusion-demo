#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch_tensorrt
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image, make_grid
from pathlib import Path

def run_inference(model_path, batch_size=4, output_dir="output", num_steps=1000):
    """
    Run inference with PyTorch model formats (TorchScript or optimized TorchScript).
    
    Args:
        model_path: Path to the model file (.ts file)
        batch_size: Batch size for inference
        output_dir: Directory to save the generated images
        num_steps: Number of diffusion steps
    """
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires CUDA.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the model
        print(f"Loading model from {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        model = model.cuda()
        print("Model loaded successfully")
        
        # Generate samples
        print(f"Generating {batch_size} samples...")
        
        # Start with random noise
        x = torch.randn(batch_size, 3, 32, 32, device="cuda")
        
        # Beta schedule
        beta_start = 0.0001
        beta_end = 0.02
        
        # Iteratively denoise
        start_time = time.time()
        for t in reversed(range(num_steps)):
            if t % 100 == 0:
                print(f"Sampling step {t}/{num_steps}")
            
            # Create timestep tensor
            timestep = torch.full((batch_size,), t, dtype=torch.int32, device="cuda")
            
            # Get model prediction
            with torch.no_grad():
                # Pass both input and timestep to the model
                predicted_noise = model(x, timestep)
            
            # Resize predicted_noise to match x if needed
            if predicted_noise.shape[2] != x.shape[2] or predicted_noise.shape[3] != x.shape[3]:
                print(f"Resizing predicted noise from {predicted_noise.shape} to {x.shape}")
                predicted_noise = F.interpolate(
                    predicted_noise, 
                    size=(x.shape[2], x.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate diffusion parameters
            beta_t = beta_start + t * (beta_end - beta_start) / (num_steps - 1)
            alpha_t = 1.0 - beta_t
            
            # Calculate alpha_bar_t (cumulative product)
            alpha_bar_t = 1.0
            for i in range(t + 1):
                alpha_i = 1.0 - (beta_start + i * (beta_end - beta_start) / (num_steps - 1))
                alpha_bar_t *= alpha_i
            
            # Calculate the mean for the reverse process step
            mean_coef1 = 1.0 / torch.sqrt(torch.tensor(alpha_t, device="cuda"))
            mean_coef2 = (1.0 - alpha_t) / torch.sqrt(torch.tensor(1.0 - alpha_bar_t, device="cuda"))
            
            mean = mean_coef1 * (x - mean_coef2 * predicted_noise)
            
            # Add noise if t > 0
            if t > 0:
                noise = torch.randn_like(x)
                noise_scale = torch.sqrt(torch.tensor(beta_t, device="cuda"))
                x = mean + noise_scale * noise
            else:
                x = mean
        
        end_time = time.time()
        print(f"Sampling complete! Total time: {end_time - start_time:.2f} seconds")
        
        # Normalize to [0, 1] range
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        
        # Save the generated images
        for i in range(batch_size):
            img_tensor = x[i].cpu()
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img.save(os.path.join(output_dir, f"sample_{i}.png"))
            print(f"Saved sample_{i}.png")
        
        # Also save a grid of all images
        try:
            grid = make_grid(x.cpu(), nrow=int(batch_size**0.5))
            save_image(grid, os.path.join(output_dir, "grid.png"))
            print(f"Saved grid of all samples to {os.path.join(output_dir, 'grid.png')}")
        except Exception as e:
            print(f"Error saving grid image: {e}")
        
        return True
    except Exception as e:
        print(f"Error running inference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run inference with PyTorch model formats")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.ts)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the generated images")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of diffusion steps")
    args = parser.parse_args()
    
    run_inference(args.model, args.batch_size, args.output_dir, args.num_steps)

if __name__ == "__main__":
    main() 