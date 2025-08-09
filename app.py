import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
import av
import asyncio
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Enhanced RTC Configuration for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

# Load model
@st.cache_resource
def load_model():
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        st.error("Model file 'model.p' not found.")
        return None

# Initialize MediaPipe
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp_hands, mp_drawing, mp_drawing_styles

# Labels dictionary
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

class RealTimeGestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.mp_hands, self.mp_drawing, self.mp_drawing_styles = init_mediapipe()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Important: False for video
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.current_prediction = "Ready for gestures..."
        self.frame_count = 0
        self.prediction_history = []
        
    def recv(self, frame):
        """Process each frame in real-time"""
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Convert BGR to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Add frame info overlay
            cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_predictions = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    H, W, _ = img.shape

                    # Extract landmark coordinates
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    # Normalize coordinates
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x - min(x_))
                        data_aux.append(landmark.y - min(y_))

                    # Make prediction
                    if self.model and len(data_aux) == 42:
                        try:
                            prediction = self.model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                            current_predictions.append(predicted_character)
                            
                            # Draw bounding box
                            x1 = int(min(x_) * W) - 10
                            y1 = int(min(y_) * H) - 10
                            x2 = int(max(x_) * W) + 10
                            y2 = int(max(y_) * H) + 10
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(img, predicted_character, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                            
                        except Exception as e:
                            cv2.putText(img, "Prediction Error", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Update prediction with smoothing
            if current_predictions:
                self.prediction_history.append(current_predictions[0])
                if len(self.prediction_history) > 10:
                    self.prediction_history = self.prediction_history[-10:]
                
                # Most common prediction in recent frames
                from collections import Counter
                most_common = Counter(self.prediction_history).most_common(1)
                self.current_prediction = most_common[0][0] if most_common else "Processing..."
            else:
                self.current_prediction = "No hands detected"
            
            # Display current prediction on frame
            cv2.putText(img, f"Current: {self.current_prediction}", (10, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            # Return error frame
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: {str(e)[:50]}", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return av.VideoFrame.from_ndarray(error_img, format="bgr24")

def main():
    st.set_page_config(
        page_title="Real-Time Gesture Recognition",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🎥 Real-Time Sign Language Recognition")
    st.markdown("**True real-time camera feed processing with live gesture detection**")
    
    # Check model availability
    model = load_model()
    if not model:
        st.error("Cannot start without model file")
        return
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Real-Time Controls")
        
        # WebRTC status
        st.subheader("📡 Connection Status")
        st.info("WebRTC streaming for real-time processing")
        
        # Performance settings
        st.subheader("🔧 Performance")
        quality = st.selectbox(
            "Video Quality",
            ["Low (320x240)", "Medium (640x480)", "High (1280x720)"],
            index=1
        )
        
        # Map quality to dimensions
        quality_map = {
            "Low (320x240)": (320, 240),
            "Medium (640x480)": (640, 480), 
            "High (1280x720)": (1280, 720)
        }
        width, height = quality_map[quality]
        
        # Gesture reference
        st.subheader("🔤 Supported Gestures")
        gesture_categories = {
            "Basic": ['I', 'YOU', 'OK', 'YES', 'NO'],
            "Actions": ['HELP', 'STOP', 'COME', 'GO'],
            "Courtesy": ['THANK YOU', 'SORRY', 'PLEASE']
        }
        
        for category, gestures in gesture_categories.items():
            with st.expander(f"{category}"):
                for gesture in gestures:
                    st.write(f"• {gesture}")
    
    # Main real-time video area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎥 Live Camera Feed")
        
        # Real-time WebRTC streamer
        ctx = webrtc_streamer(
            key="real-time-gesture-recognition",
            video_transformer_factory=RealTimeGestureTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"exact": width},
                    "height": {"exact": height},
                    "frameRate": {"min": 15, "ideal": 30, "max": 30}
                },
                "audio": False
            },
            async_processing=True,
            video_html_attrs={
                "style": {
                    "width": "100%",
                    "border": "3px solid #00ff00",
                    "border-radius": "10px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.1)"
                }
            }
        )
        
        # Real-time status
        if ctx.video_transformer:
            st.success("✅ Real-time processing active")
        else:
            st.warning("⏳ Initializing real-time stream...")
    
    with col2:
        st.subheader("📊 Live Status")
        
        # Real-time prediction display
        prediction_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Update display in real-time
        if ctx.video_transformer:
            current_pred = getattr(ctx.video_transformer, 'current_prediction', 'Initializing...')
            frame_count = getattr(ctx.video_transformer, 'frame_count', 0)
            
            if current_pred and current_pred != "No hands detected":
                prediction_placeholder.success(f"**🎯 Detected:** {current_pred}")
            else:
                prediction_placeholder.info(f"**📱 Status:** {current_pred}")
            
            status_placeholder.metric("Frames Processed", frame_count)
        
        # Instructions
        st.subheader("💡 Tips")
        st.write("""
        **For best real-time results:**
        
        🔆 **Good lighting** on hands
        
        📱 **Hold steady** for 2-3 seconds
        
        🎯 **Clear gestures** work best
        
        🖐️ **One hand** at a time initially
        
        📏 **Appropriate distance** from camera
        """)
        
        # Performance info
        st.subheader("⚡ Performance")
        st.write(f"**Resolution:** {width}x{height}")
        st.write(f"**Target FPS:** 30")
        st.write(f"**Processing:** Real-time")

if __name__ == "__main__":
    main()
