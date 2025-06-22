"""
Test script to verify the models and app components work correctly
"""

import torch
from model import Generator, Discriminator
import matplotlib.pyplot as plt

def test_models():
    """Test that models can be created and run"""
    print("Testing model creation...")
    
    # Test Generator
    generator = Generator(noise_dim=100, num_classes=10)
    noise = torch.randn(5, 100)
    labels = torch.randint(0, 10, (5,))
    
    with torch.no_grad():
        fake_images = generator(noise, labels)
    
    print(f"Generator output shape: {fake_images.shape}")
    assert fake_images.shape == (5, 1, 28, 28), "Generator output shape incorrect"
    
    # Test Discriminator
    discriminator = Discriminator(num_classes=10)
    with torch.no_grad():
        output = discriminator(fake_images, labels)
    
    print(f"Discriminator output shape: {output.shape}")
    assert output.shape == (5, 1), "Discriminator output shape incorrect"
    
    print("✅ Model tests passed!")

def test_image_generation():
    """Test image generation for all digits"""
    print("\nTesting image generation for all digits...")
    
    generator = Generator(noise_dim=100, num_classes=10)
    generator.eval()
    
    # Generate one image for each digit
    with torch.no_grad():
        noise = torch.randn(10, 100)
        labels = torch.arange(0, 10)
        fake_images = generator(noise, labels)
        
        # Denormalize
        fake_images = fake_images * 0.5 + 0.5
        fake_images = torch.clamp(fake_images, 0, 1)
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(fake_images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'Digit {i}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_generation.png')
    plt.show()
    print("✅ Image generation test completed! Check 'test_generation.png'")

def test_streamlit_components():
    """Test that Streamlit app can import all components"""
    print("\nTesting Streamlit app imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        from model import Generator
        print("✅ Model imported successfully")
        
        import torch
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
        print("✅ All dependencies imported successfully")
        
        print("🎉 All components ready for Streamlit app!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all requirements are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    print("🧪 Running comprehensive tests...\n")
    
    test_models()
    test_image_generation()
    test_streamlit_components()
    
    print("\n🎉 All tests completed!")
    print("\nNext steps:")
    print("1. Train the model in Google Colab using train_model.py")
    print("2. Download the trained generator_model.pth")
    print("3. Deploy on Streamlit Cloud")
    print("4. Or run locally with: streamlit run app.py")
