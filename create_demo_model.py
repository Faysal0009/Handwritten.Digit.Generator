"""
Create a sample trained model for demonstration purposes.
This script creates a basic trained model that can generate simple digit-like patterns.
In practice, you should train the model properly using train_model.py in Google Colab.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Generator, Discriminator

def create_demo_model():
    """Create a minimally trained model for demonstration"""
    device = torch.device('cpu')
    
    # Initialize generator
    generator = Generator(noise_dim=100, num_classes=10)
    
    # Load MNIST data for a few training steps
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Quick training for demonstration (just a few batches)
        discriminator = Discriminator(num_classes=10)
        criterion = nn.BCELoss()
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        generator.train()
        discriminator.train()
        
        print("Training demo model for a few iterations...")
        
        # Train for just 2 epochs with limited batches
        for epoch in range(2):
            for i, (real_images, real_labels) in enumerate(train_loader):
                if i >= 20:  # Only train on first 20 batches per epoch
                    break
                    
                batch_size = real_images.size(0)
                
                # Train discriminator
                optimizer_D.zero_grad()
                real_target = torch.ones(batch_size, 1)
                fake_target = torch.zeros(batch_size, 1)
                
                real_output = discriminator(real_images, real_labels)
                real_loss = criterion(real_output, real_target)
                
                noise = torch.randn(batch_size, 100)
                fake_labels = torch.randint(0, 10, (batch_size,))
                fake_images = generator(noise, fake_labels)
                fake_output = discriminator(fake_images.detach(), fake_labels)
                fake_loss = criterion(fake_output, fake_target)
                
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_D.step()
                
                # Train generator
                optimizer_G.zero_grad()
                fake_output = discriminator(fake_images, fake_labels)
                g_loss = criterion(fake_output, real_target)
                g_loss.backward()
                optimizer_G.step()
                
                if i % 10 == 0:
                    print(f'Epoch {epoch+1}, Batch {i+1}, D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Save the model
        torch.save(generator.state_dict(), 'generator_model.pth')
        print("Demo model created and saved as 'generator_model.pth'")
        
    except Exception as e:
        print(f"Could not create demo model: {e}")
        print("The app will work with an untrained model for demonstration.")

if __name__ == "__main__":
    create_demo_model()
