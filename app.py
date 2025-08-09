import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Webcam Stream", page_icon="📷")

st.title("📷 Live Webcam in Streamlit")

# Video processing class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to OpenCV format
        
        # Example: Draw text on video
        cv2.putText(img, "Live Stream", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Create webcam stream
webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.write("If you see a black screen, allow camera access in your browser settings.")
