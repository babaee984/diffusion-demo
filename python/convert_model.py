#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# Try to import torch_tensorrt for optimized models
try:
    import torch_tensorrt
    TORCH_TENSORRT_AVAILABLE = True
except ImportError:
    TORCH_TENSORRT_AVAILABLE = False
    print("Warning: torch_tensorrt not available. Optimized conversion will not be possible.")

# Add the parent directory to the path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.model import UNet

def load_model(model_path, device="cuda"):
    """
    Load a PyTorch model from a file.
    
    Args:
        model_path: Path to the PyTorch model (.pth file)
        device: Device to load the model to ("cuda" or "cpu")
        
    Returns:
        The loaded model
    """
    print(f"Loading PyTorch model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the checkpoint contains UNet model
    if isinstance(checkpoint, dict) and "time_embed.weight" in checkpoint:
        print("Detected UNet model checkpoint")
        # Create UNet model and load state dict directly
        model = UNet()
        model.load_state_dict(checkpoint)
    else:
        print("Checkpoint format not recognized")
        return None
    
    model.eval()
    model = model.to(device)
    print("Model loaded successfully")
    return model

def convert_to_torchscript(model_path, output_path, batch_size=4, image_size=32, optimize=False, precision="fp16"):
    """
    Convert a PyTorch model to TorchScript format with optional optimization.
    
    Args:
        model_path: Path to the PyTorch model (.pth file)
        output_path: Path to save the TorchScript model (.ts file)
        batch_size: Batch size for inference
        image_size: Size of the input images (assumed square)
        optimize: Whether to apply TensorRT optimization
        precision: Precision to use for optimization ("fp32", "fp16", or "int8")
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU for conversion.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check for torch_tensorrt if optimization is requested
    if optimize and not TORCH_TENSORRT_AVAILABLE:
        print("Error: Optimization requested but torch_tensorrt is not available.")
        print("Please install torch_tensorrt or set optimize=False.")
        return False
    
    if optimize and TORCH_TENSORRT_AVAILABLE:
        print(f"torch_tensorrt version: {torch_tensorrt.__version__}")
    
    try:
        # Load the model
        model = load_model(model_path, device)
        if model is None:
            return False
        
        # Create sample inputs for tracing
        example_input = torch.randn(batch_size, 3, image_size, image_size, device=device)
        example_timestep = torch.full((batch_size,), 500, dtype=torch.int32, device=device)
        
        # Trace the model
        print("Tracing the model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (example_input, example_timestep))
            traced_model = torch.jit.freeze(traced_model)
        print("Model traced successfully")
        
        # Apply optimization if requested
        if optimize and TORCH_TENSORRT_AVAILABLE and device == "cuda":
            print(f"Applying optimization with precision: {precision}")
            
            # Set precision
            enabled_precisions = {torch.float32}
            if precision == "fp16":
                enabled_precisions = {torch.float16}
            elif precision == "int8":
                enabled_precisions = {torch.int8}
            
            # Compile with TensorRT
            print(f"Compiling model with TensorRT using {precision} precision...")
            trt_model = torch_tensorrt.compile(
                traced_model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=[1, 3, image_size, image_size],
                        opt_shape=[batch_size, 3, image_size, image_size],
                        max_shape=[batch_size*2, 3, image_size, image_size],
                        dtype=torch.float32
                    ),
                    torch_tensorrt.Input(
                        min_shape=[1],
                        opt_shape=[batch_size],
                        max_shape=[batch_size*2],
                        dtype=torch.int32
                    )
                ],
                enabled_precisions=enabled_precisions,
                workspace_size=1 << 30,  # 1GB workspace size
                require_full_compilation=False  # Allow PyTorch fallback for unsupported ops
            )
            final_model = trt_model
            print("Optimization applied successfully")
        else:
            final_model = traced_model
            if optimize and device != "cuda":
                print("Optimization skipped: CUDA not available")
            elif optimize and not TORCH_TENSORRT_AVAILABLE:
                print("Optimization skipped: torch_tensorrt not available")
            else:
                print("No optimization requested, using standard TorchScript")
        
        # Save the model
        print(f"Saving model to {output_path}")
        torch.jit.save(final_model, output_path)
        print("Model saved successfully")
        
        # Test the model
        print("Testing the model...")
        test_input = torch.randn(1, 3, image_size, image_size, device=device)
        test_timestep = torch.full((1,), 500, dtype=torch.int32, device=device)
        with torch.no_grad():
            result = final_model(test_input, test_timestep)
        print(f"Test successful! Output shape: {result.shape}")
        
        return True
    except Exception as e:
        print(f"Error converting model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to TorchScript format with optional optimization")
    parser.add_argument("--input", type=str, required=True, help="Path to the PyTorch model (.pth file)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the TorchScript model (.ts file)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the model")
    parser.add_argument("--image-size", type=int, default=32, help="Size of the input images (assumed square)")
    parser.add_argument("--optimize", action="store_true", help="Whether to apply TensorRT optimization")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], 
                        help="Precision to use for optimization (only used if --optimize is set)")
    
    args = parser.parse_args()
    
    # Ensure output path has .ts extension
    output_path = args.output
    if not output_path.endswith('.ts'):
        output_path = f"{output_path}.ts"
        print(f"Adding .ts extension to output path: {output_path}")
    
    # Convert the model
    convert_to_torchscript(
        args.input, 
        output_path, 
        args.batch_size, 
        args.image_size, 
        args.optimize, 
        args.precision
    )

if __name__ == "__main__":
    main() 