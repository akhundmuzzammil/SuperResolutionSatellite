import streamlit as st
from PIL import Image
import time
import numpy as np
from pathlib import Path
from model import get_model
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Satellite Image Super Resolution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
    }
    .title {
        text-align: center;
        color: #2c3e50;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

# At the start of the file, after imports, add:
if 'temp_file_path' not in st.session_state:
    st.session_state['temp_file_path'] = None

# Title and description
st.markdown("<h1 class='title'>🛰️ Satellite Image Super Resolution</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Enhance your satellite imagery using AI</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/WarrenGreen/srcnn/master/results/05261_combined.jpg", 
             caption="Sample Enhancement")
    st.markdown("## About")
    st.markdown("""
    This app uses a Super Resolution Convolutional Neural Network (SRCNN) 
    to enhance the resolution of satellite imagery.
    
    ### Features
    - Real-time image enhancement
    - Support for various image formats
    - Advanced AI-powered upscaling
    
    ### How to use
    1. Upload your satellite image
    2. Click enhance
    3. Download the result
    """)

# Main content
def process_image(image):
    # Convert to numpy array and normalize
    img_array = np.asarray(image, dtype="uint8")
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Load model and predict
    model = get_model("models/weights2.h5")
    enhanced = model.predict(img_array)
    
    # Post-process
    enhanced = np.clip(enhanced[0] * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced)

# File uploader
uploaded_file = st.file_uploader("Choose a satellite image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center'>Original Image</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        # Resize if image is too large
        if image.size[0] > 400 or image.size[1] > 400:
            image = image.resize((400, 400))
        st.image(image, use_container_width=True)

    # Enhance button
    if st.button("🚀 Enhance Image"):
        with st.spinner('Enhancing your image...'):
            # Progress bar animation
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            # Process image
            enhanced_image = process_image(image)
            
            # Display enhanced image
            with col2:
                st.markdown("<h3 style='text-align: center'>Enhanced Image</h3>", 
                          unsafe_allow_html=True)
                st.image(enhanced_image, use_container_width=True)
                
                # Clean up previous temp file if it exists
                if st.session_state['temp_file_path'] and os.path.exists(st.session_state['temp_file_path']):
                    try:
                        os.unlink(st.session_state['temp_file_path'])
                    except:
                        pass

                # Create new temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp_file.close()  # Close the file immediately
                enhanced_image.save(temp_file.name)
                st.session_state['temp_file_path'] = temp_file.name
                
                with open(temp_file.name, 'rb') as file:
                    if st.download_button(
                        label="Download Enhanced Image",
                        data=file,
                        file_name="enhanced_satellite_image.png",
                        mime="image/png"
                    ):
                        # Only try to delete the file after successful download
                        try:
                            os.unlink(st.session_state['temp_file_path'])
                            st.session_state['temp_file_path'] = None
                        except:
                            pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit • Powered by SRCNN</p>
</div>
""", unsafe_allow_html=True)
