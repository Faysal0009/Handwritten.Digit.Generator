import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
from model import Generator

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="wide"
)


@st.cache_resource
def load_model():
    device = torch.device('cpu')  
    generator = Generator(noise_dim=100, num_classes=10)
    
    try:
       
        generator.load_state_dict(torch.load('generator_model.pth', map_location=device))
        generator.eval()
        return generator
    except FileNotFoundError:
        
        generator.eval()
        return generator

def generate_digit_images(generator, digit, num_images=5):
    """Generate specified number of images for a given digit"""
    device = torch.device('cpu')
    
    with torch.no_grad():
       
        noise = torch.randn(num_images, 100)
        labels = torch.full((num_images,), digit, dtype=torch.long)
        
        fake_images = generator(noise, labels)
        
       
        fake_images = fake_images * 0.5 + 0.5
        fake_images = torch.clamp(fake_images, 0, 1)
        
        return fake_images

def display_images(images, digit):
    """Display generated images in a grid"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle(f'Generated Images for Digit {digit}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    st.title("🔢 Handwritten Digit Generator")
    st.markdown("Generate realistic handwritten digits using a trained Conditional GAN!")
    
 
    generator = load_model()
    
    
    st.header("Generate Handwritten Digits")
    
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        selected_digit = st.selectbox(
            "Select digit to generate:",
            options=list(range(10)),
            index=0
        )
    
    with col2:
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Digit", use_container_width=True):
            import random
            selected_digit = random.randint(0, 9)
            st.success(f"Random digit selected: {selected_digit}")
    
    st.markdown(f"### Generating Digit: **{selected_digit}**")
    
    if st.button("Generate 5 Images", type="primary"):
        with st.spinner("Generating images..."):
           
            generated_images = generate_digit_images(generator, selected_digit, 5)
            
           
            fig = display_images(generated_images, selected_digit)
            st.pyplot(fig)
            
            
            st.subheader("Individual Images (28x28 pixels)")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                with col:
                    
                    img_array = generated_images[i].squeeze().numpy()
                    img_array = (img_array * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array, mode='L')
                    
                   
                    img_resized = img_pil.resize((140, 140), Image.NEAREST)
                    
                    st.image(img_resized, caption=f"Sample {i+1}")

  
    st.markdown("---")
    st.markdown("Built with Streamlit and PyTorch")

if __name__ == "__main__":
    main()
