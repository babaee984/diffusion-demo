# Diffusion Model from Scratch

This repository contains an implementation of a Denoising Diffusion Probabilistic Model (DDPM) with various deployment options for inference. The model is trained on CIFAR-10 dataset (32x32 images) and can be deployed using different methods:

1. PyTorch (original model)
2. TorchScript (traced model)
3. Optimized TorchScript (enhanced performance)

## Project Structure

```
.
├── python/                 # Python implementation of the diffusion model
│   ├── model.py            # Model architecture (UNet, DDPM)
│   ├── train.py            # Training script
│   ├── run_inference.py    # Python script for inference (supports all model types)
│   ├── convert_model.py    # Script to convert PyTorch to TorchScript (with optional optimization)
│   └── ...
├── models/                 # Saved models
│   ├── ddpm.pth            # Original PyTorch model
│   ├── ddpm.ts             # TorchScript model
│   └── ddpm_optimized.ts   # Optimized TorchScript model
└── README.md               # This file
```

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+
- TensorRT 8.0+ (for optimized models)
- torch-tensorrt (for PyTorch optimized conversion)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Diffusion-from-scratch.git
cd Diffusion-from-scratch
```

2. Install Python dependencies:
```bash
pip install torch torchvision tqdm pillow matplotlib
pip install torch-tensorrt  # For optimized TorchScript models
```

## Model Conversion

### Converting PyTorch Models

Our unified conversion script supports both standard TorchScript and optimized TorchScript conversion:

#### Standard TorchScript Conversion:

```bash
python python/convert_model.py --input models/ddpm.pth --output models/ddpm.ts --batch-size 4 --image-size 32
```

#### Optimized TorchScript Conversion:

```bash
python python/convert_model.py --input models/ddpm.pth --output models/ddpm_optimized.ts --batch-size 4 --image-size 32 --optimize --precision fp16
```

Available precision options (when using `--optimize`):
- `fp32`: Full precision (slower but most accurate)
- `fp16`: Half precision (good balance of speed and accuracy)
- `int8`: Integer quantization (fastest but may reduce quality)

## Running Inference

### Python Inference with Any Model Type

Our unified inference script supports all model types (TorchScript and optimized TorchScript):

```bash
python python/run_inference.py --model models/ddpm_optimized.ts --batch-size 4 --output-dir output
```

You can use this script with any of the model formats by simply changing the model path:

- For TorchScript: `--model models/ddpm.ts`
- For Optimized TorchScript: `--model models/ddpm_optimized.ts`

## Model Comparison

| Format | Advantages | Disadvantages | Use Case |
|--------|------------|---------------|----------|
| PyTorch | - Easy to use<br>- Flexible | - Slower inference<br>- Larger file size | Development, research |
| TorchScript | - Portable<br>- Language agnostic<br>- Optimized | - Less flexible than PyTorch | Production deployment |
| Optimized TorchScript | - Combines TorchScript portability with hardware optimization<br>- Best performance | - NVIDIA GPUs only<br>- Requires torch-tensorrt | Best option for NVIDIA GPU deployment |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or image size.
2. **Model compatibility**: Ensure your model operations are supported by the optimization level you're using.
3. **Dimension mismatch**: Check input/output dimensions in your model.
4. **CUDA errors**: Make sure your CUDA version is compatible with the PyTorch version you're using.
5. **Model loading errors**: Ensure the model was properly exported to TorchScript or optimized format.
6. **Optimization errors**: Check that your GPU supports the precision mode you selected for optimization.

### Handling Dimension Mismatches

The current implementation handles dimension mismatches between the model output (32x32) and the expected input (64x64) by automatically resizing the predicted noise using bilinear interpolation.

## Customization

- To train on a different dataset, modify the `train.py` script.
- To change the model architecture, modify the `model.py` file.
- To adjust inference parameters, modify the batch size, number of steps, or precision in the respective scripts.

## Performance Comparison

Here's a comparison of inference times for generating 2 images with different deployment methods:

| Method | Time (seconds) | Speedup |
|--------|----------------|---------|
| PyTorch | ~5.0 | 1.0x |
| TorchScript | ~1.3 | 3.8x |
| Optimized TorchScript | ~0.7 | 7.1x |

*Note: These are approximate values and may vary depending on your hardware and configuration.*

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The diffusion model implementation is based on [DDPM paper](https://arxiv.org/abs/2006.11239).
- Optimization support is based on [Torch-TensorRT](https://github.com/pytorch/TensorRT).
