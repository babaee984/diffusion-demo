from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from model import UNet, DDPM
from torch.cuda.amp import autocast, GradScaler
import os

from torchvision.utils import save_image

def save_images(images, filename, nrow=4):
    """
    Save a batch of generated images to a file.

    Args:
        images (torch.Tensor): Batch of images, shape [N, C, H, W], range [-1, 1].
        filename (str): Output file path (e.g., "epoch_1.png").
        nrow (int): Number of images per row in the grid (default: 4).
    """
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    # Clamp to ensure valid pixel values
    images = torch.clamp(images, 0, 1)
    # Save as a grid
    save_image(images, filename, nrow=nrow)

# Data transformation for CIFAR-10
# Important: Resize to 32x32 to match the model architecture
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
])

# Use CIFAR-10 dataset (3 channels - RGB)
print("Loading CIFAR-10 dataset...")
cifar_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

# Create data loader
loader = DataLoader(cifar_train, batch_size=32, shuffle=True)

# Initialize model, optimizer, and scaler
print("Initializing model...")
model = UNet().cuda()
ddpm = DDPM(model).cuda()
optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
scaler = GradScaler()

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Training loop
print("Starting training...")
for epoch in range(50):  # ~1-2 days on RTX 3080
    total_loss = 0.0
    num_batches = 0
    
    for x, _ in loader:
        # CIFAR-10 is 32x32, but model expects 64x64, so resize
        
        x = x.cuda()
        
        optimizer.zero_grad()
        with autocast():
            loss = ddpm(x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Generate and save samples
    samples = ddpm.sample(16)
    save_images(samples, f"epoch_{epoch}.png")
    
    # Save model checkpoint every 5 epochs
    if epoch % 5 == 0 or epoch == 49:
        torch.save(model.state_dict(), f"models/ddpm_epoch_{epoch}.pth")

# Save final model
torch.save(model.state_dict(), "models/ddpm.pth")
print("Training completed!")


