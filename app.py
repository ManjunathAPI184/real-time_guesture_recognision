import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Do your processing here
        return img

st.title("Live Webcam with Streamlit Cloud")
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
