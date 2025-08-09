import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import av
import logging
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Enhanced configuration specifically for Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ]
})

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.mp_hands, self.mp_drawing, self.mp_drawing_styles = init_mediapipe()
        self.hands = None
        self.confidence = 0.3
        self.current_prediction = "Initializing camera..."
        self.frame_count = 0
        
    def transform(self, frame):
        try:
            self.frame_count += 1
            
            # Convert frame to numpy array
            img = frame.to_ndarray(format="bgr24")
            
            # Add frame counter overlay to verify stream is working
            cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Initialize MediaPipe if needed
            if self.hands is None:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.confidence,
                    min_tracking_confidence=0.5
                )
            
            # Your existing gesture recognition code here...
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Rest of your processing logic...
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Log error and return a colored frame to show stream is active
            error_img = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)  # Red frame
            cv2.putText(error_img, f"Error: {str(e)[:50]}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(error_img, format="bgr24")

# In your main function, use this enhanced WebRTC configuration:
def main():
    st.title("🤟 Real-Time Sign Language Recognition")
    
    # Add debugging info
    st.info("📹 Camera Status: If you see a blank screen, try the solutions below")
    
    # WebRTC streamer with enhanced settings
    try:
        ctx = webrtc_streamer(
            key="gesture-detection-v2",  # Changed key to force new connection
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "max": 1280, "ideal": 640},
                    "height": {"min": 240, "max": 720, "ideal": 480},
                    "frameRate": {"min": 15, "max": 30, "ideal": 25},
                    "facingMode": "user"  # Use front camera
                },
                "audio": False
            },
            async_processing=True,
            video_html_attrs={
                "style": {
                    "width": "100%",
                    "max-width": "640px",
                    "margin": "0 auto",
                    "border": "3px solid #00ff00",
                    "border-radius": "10px",
                    "background-color": "#000000"
                },
                "controls": False,
                "autoplay": True,
                "muted": True
            }
        )
        
    except Exception as e:
        st.error(f"WebRTC failed to initialize: {str(e)}")
