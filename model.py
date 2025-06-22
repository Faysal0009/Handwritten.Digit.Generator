import torch
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

def create_models():
    """Create and return generator and discriminator models"""
    generator = Generator()
    discriminator = Discriminator()
    return generator, discriminator
