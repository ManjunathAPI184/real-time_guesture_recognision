import streamlit as st
from PIL import Image
import io

st.title("Live Webcam Capture in Streamlit Cloud")

# Capture from webcam
img_data = st.camera_input("Take a picture")

if img_data:
    # Convert to PIL image
    image = Image.open(io.BytesIO(img_data.getvalue()))
    st.image(image, caption="Captured Image", use_container_width=True)
