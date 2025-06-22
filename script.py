# ============================================================================
# COPY-PASTE TRAINING SCRIPT FOR GOOGLE COLAB
# ============================================================================
# Instructions:
# 1. Go to https://colab.research.google.com
# 2. Create new notebook
# 3. Change Runtime → Runtime type → T4 GPU
# 4. Copy-paste this entire script into a code cell
# 5. Run the cell (it will take 30-45 minutes)
# 6. Download generator_model.pth when done
# ============================================================================

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: GPU not available! Please change runtime to T4 GPU")

# Install packages if needed
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import torch
    import torchvision
    import matplotlib
    print("✅ All packages already installed")
except ImportError:
    print("Installing required packages...")
    install_package("torch")
    install_package("torchvision")
    install_package("matplotlib")

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Generator network
        self.fc1 = nn.Linear(noise_dim + 50, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28 * 28)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, noise, labels):
        # Embed labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate noise and label embedding
        x = torch.cat([noise, label_embed], dim=1)
        
        # Forward pass
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = torch.tanh(self.fc4(x))
        
        # Reshape to image format
        x = x.view(-1, 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Discriminator network
        self.fc1 = nn.Linear(28 * 28 + 50, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, images, labels):
        # Flatten images
        images = images.view(-1, 28 * 28)
        
        # Embed labels
        label_embed = self.label_embedding(labels)
        
        # Concatenate image and label embedding
        x = torch.cat([images, label_embed], dim=1)
        
        # Forward pass
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        
        return x

print("✅ Model classes defined")

# ============================================================================
# TRAINING SETUP
# ============================================================================

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters - ULTRA FAST FOR 40-MINUTE DEADLINE
BATCH_SIZE = 256  # Increased for speed
LEARNING_RATE = 0.0003  # Higher for faster convergence
NUM_EPOCHS = 12  # Reduced to 12 for ultra-fast training
NOISE_DIM = 100
NUM_CLASSES = 10

print(f"🚨 ULTRA-FAST Configuration: {NUM_EPOCHS} epochs, ~{NUM_EPOCHS * 1.2:.0f} minutes training time")

# Load MNIST dataset
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

print(f"✅ Dataset loaded: {len(train_dataset)} samples, {len(train_loader)} batches")

# Initialize models
generator = Generator(NOISE_DIM, NUM_CLASSES).to(device)
discriminator = Discriminator(NUM_CLASSES).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

print(f"✅ Models initialized on {device}")
print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

def save_sample_images(epoch):
    """Save sample generated images"""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(10, NOISE_DIM).to(device)
        labels = torch.arange(0, 10).to(device)
        fake_images = generator(noise, labels)
        fake_images = fake_images * 0.5 + 0.5
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_images[i].cpu().squeeze(), cmap='gray')
            axes[row, col].set_title(f'Digit {i}')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Generated Images - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        plt.show()
    generator.train()

print("🚀 Starting training...")
print("=" * 60)

# Training loop
generator.train()
discriminator.train()

g_losses = []
d_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_g_loss = 0
    epoch_d_loss = 0
    
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
        fake_output = discriminator(fake_images, fake_labels)
        g_loss = criterion(fake_output, real_target)
        g_loss.backward()
        optimizer_G.step()
        
        # Accumulate losses
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        
        # Print progress
        if i % 150 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                  f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
    
    # Track average losses
    avg_g_loss = epoch_g_loss / len(train_loader)
    avg_d_loss = epoch_d_loss / len(train_loader)
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    
    print(f'✅ Epoch [{epoch+1}/{NUM_EPOCHS}] - Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}')
    
    # Show sample images every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_sample_images(epoch + 1)

print("\n🎉 Training completed!")

# ============================================================================
# SAVE MODEL AND FINAL TESTS
# ============================================================================

# Save the trained model
torch.save(generator.state_dict(), 'generator_model.pth')
print("✅ Model saved as 'generator_model.pth'")

# Plot training losses
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(g_losses, label='Generator Loss', color='blue')
plt.plot(d_losses, label='Discriminator Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Losses')
plt.legend()
plt.grid(True)

# Final test - generate 5 images of digit 7
plt.subplot(1, 2, 2)
generator.eval()
with torch.no_grad():
    noise = torch.randn(5, NOISE_DIM).to(device)
    labels = torch.full((5,), 7).to(device)
    test_images = generator(noise, labels)
    test_images = test_images * 0.5 + 0.5
    
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(test_images[i].cpu().squeeze(), cmap='gray')
        plt.title(f'Gen 7 #{i+1}')
        plt.axis('off')

plt.tight_layout()
plt.show()

# Generate final test for all digits
print("🎯 Final test: All digits")
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

with torch.no_grad():
    for digit in range(10):
        noise = torch.randn(2, NOISE_DIM).to(device)
        labels = torch.full((2,), digit).to(device)
        digit_images = generator(noise, labels)
        digit_images = digit_images * 0.5 + 0.5
        
        for i in range(2):
            axes[i, digit].imshow(digit_images[i].cpu().squeeze(), cmap='gray')
            axes[i, digit].set_title(f'Digit {digit}')
            axes[i, digit].axis('off')

plt.suptitle('Final Generated Images - All Digits', fontsize=16)
plt.tight_layout()
plt.show()

print(f"\n📊 Training Summary:")
print(f"- Final Generator Loss: {g_losses[-1]:.4f}")
print(f"- Final Discriminator Loss: {d_losses[-1]:.4f}")
print(f"- Total epochs: {NUM_EPOCHS}")
print(f"- Model size: ~2MB")

# Download the model
print("\n📥 Downloading model...")
try:
    from google.colab import files
    files.download('generator_model.pth')
    print("✅ Download started! Check your browser's downloads folder.")
except:
    print("💡 To download: Click on files panel (left) → right-click 'generator_model.pth' → Download")

print("\n🎯 Next Steps:")
print("1. ✅ Training completed")
print("2. 📥 Download 'generator_model.pth'")
print("3. 📤 Upload to your GitHub repository")
print("4. 🚀 Deploy on Streamlit Cloud")
print("\n🎉 Great job! Your model is ready for deployment!")
