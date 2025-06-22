"""
MNIST Digit Generation Training Script
This script trains a Conditional GAN (cGAN) to generate handwritten digits.

Instructions for Google Colab:
1. Upload this file to Google Colab
2. Make sure GPU is enabled (Runtime -> Change runtime type -> GPU -> T4)
3. Run all cells
4. The trained model will be saved as 'generator_model.pth'
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import Generator, Discriminator

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS = 50
NOISE_DIM = 100
NUM_CLASSES = 10

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

# Initialize models
generator = Generator(NOISE_DIM, NUM_CLASSES).to(device)
discriminator = Discriminator(NUM_CLASSES).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Training function
def train_gan():
    generator.train()
    discriminator.train()
    
    for epoch in range(NUM_EPOCHS):
        for i, (real_images, real_labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            
            # Labels for real and fake data
            real_target = torch.ones(batch_size, 1).to(device)
            fake_target = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real data
            real_output = discriminator(real_images, real_labels)
            real_loss = criterion(real_output, real_target)
            
            # Fake data
            noise = torch.randn(batch_size, NOISE_DIM).to(device)
            fake_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
            fake_images = generator(noise, fake_labels)
            fake_output = discriminator(fake_images.detach(), fake_labels)
            fake_loss = criterion(fake_output, fake_target)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate fake images and get discriminator output
            fake_output = discriminator(fake_images, fake_labels)
            g_loss = criterion(fake_output, real_target)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Print progress
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(epoch + 1)
    
    # Save the trained generator
    torch.save(generator.state_dict(), 'generator_model.pth')
    print("Training completed! Model saved as 'generator_model.pth'")

def save_sample_images(epoch):
    """Save sample generated images"""
    generator.eval()
    with torch.no_grad():
        # Generate one image for each digit
        noise = torch.randn(10, NOISE_DIM).to(device)
        labels = torch.arange(0, 10).to(device)
        fake_images = generator(noise, labels)
        
        # Denormalize images
        fake_images = fake_images * 0.5 + 0.5
        
        # Create subplot
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_images[i].cpu().squeeze(), cmap='gray')
            axes[row, col].set_title(f'Digit {i}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'sample_epoch_{epoch}.png')
        plt.show()
    
    generator.train()

if __name__ == "__main__":
    print("Starting training...")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters())}")
    
    train_gan()
