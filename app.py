import streamlit as st
import cv2
import numpy as np

st.title("Live Webcam Feed in Streamlit Cloud")

# Capture image from webcam via browser
img_file = st.camera_input("Take a photo or live snapshot")

if img_file is not None:
    # Convert the image to OpenCV format
    bytes_data = img_file.getvalue()
    img_array = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Process the frame if needed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show the frame
    st.image(frame, channels="BGR")
    st.image(gray, channels="GRAY")
