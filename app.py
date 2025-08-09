import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import av
import logging
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Enhanced WebRTC configuration for Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        st.error("Model file 'model.p' not found. Please ensure the model file is in the same directory.")
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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.mp_hands, self.mp_drawing, self.mp_drawing_styles = init_mediapipe()
        self.hands = None
        self.confidence = 0.3
        self.current_prediction = "No gesture detected"
        self.frame_count = 0
        
    def update_confidence(self, confidence):
        self.confidence = confidence
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )
    
    def transform(self, frame):
        try:
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            
            # Initialize hands if not done
            if self.hands is None:
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.confidence,
                    min_tracking_confidence=0.5
                )
            
            # Process the frame
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Reset prediction
            self.current_prediction = f"Processing frame {self.frame_count}"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    H, W, _ = img.shape

                    # Extract landmark coordinates
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    # Normalize coordinates
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    # Bounding box coordinates
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    # Make prediction
                    if self.model and len(data_aux) == 42:
                        try:
                            prediction = self.model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                            self.current_prediction = predicted_character
                            
                            # Draw bounding box and text
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.putText(img, predicted_character, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        except Exception as e:
                            self.current_prediction = f"Prediction error: {str(e)[:20]}"

                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            else:
                self.current_prediction = "No hands detected"
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logging.error(f"Transform error: {str(e)}")
            # Return original frame if processing fails
            return frame

def main():
    st.title("🤟 Real-Time Sign Language Recognition System")
    st.markdown("---")
    
    # Add connection status
    st.info("🔄 **Status**: Initializing WebRTC connection...")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if model.p exists.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    detection_confidence = st.sidebar.slider(
        "Detection Confidence", 
        0.1, 1.0, 0.3, 0.1,
        help="Higher values require more confident detections"
    )
    
    # Display available gestures
    st.sidebar.subheader("🔤 Available Gestures")
    col1, col2 = st.sidebar.columns(2)
    gestures_list = list(labels_dict.values())
    mid = len(gestures_list) // 2
    
    with col1:
        for gesture in gestures_list[:mid]:
            st.sidebar.write(f"• {gesture}")
    
    with col2:
        for gesture in gestures_list[mid:]:
            st.sidebar.write(f"• {gesture}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Real-Time Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.subheader("Real-Time Sign Language Detection")
        st.write("Real-time gesture recognition using your webcam through WebRTC streaming.")
        
        # Troubleshooting info
        st.warning("""
        **If you see a blank screen:**
        1. Allow camera permissions when prompted
        2. Try refreshing the page
        3. Check if your browser supports WebRTC
        4. Ensure stable internet connection
        5. Try a different browser (Chrome recommended)
        """)
        
        # Instructions
        st.info("""
        **Instructions:**
        1. Click 'START' to begin real-time detection
        2. Allow camera permissions when prompted
        3. Show clear hand gestures to the camera
        4. The prediction will appear above your hand in real-time
        5. Adjust detection confidence in the sidebar if needed
        """)
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC streamer with enhanced configuration
            try:
                ctx = webrtc_streamer(
                    key="sign-language-detection",
                    mode=WebRtcMode.SENDRECV,
                    video_transformer_factory=VideoTransformer,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 320, "ideal": 640, "max": 1280},
                            "height": {"min": 240, "ideal": 480, "max": 720},
                            "frameRate": {"min": 15, "ideal": 30, "max": 30}
                        },
                        "audio": False
                    },
                    async_processing=True,
                    video_html_attrs={
                        "style": {
                            "width": "100%", 
                            "margin": "0 auto", 
                            "border": "2px solid #1f77b4",
                            "border-radius": "10px"
                        },
                        "controls": False,
                        "autoplay": True,
                        "muted": True
                    }
                )
                
                # Update confidence in real-time
                if ctx.video_transformer:
                    ctx.video_transformer.update_confidence(detection_confidence)
                    
            except Exception as e:
                st.error(f"WebRTC initialization failed: {str(e)}")
                st.info("Falling back to camera input method...")
                
                # Fallback to camera input
                st.subheader("📸 Alternative: Camera Capture")
                camera_photo = st.camera_input("Take a photo for gesture detection")
                
                if camera_photo is not None:
                    # Process camera photo (your existing image processing code)
                    image = Image.open(camera_photo)
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Initialize MediaPipe for static image
                    mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
                    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=detection_confidence)
                    
                    # Process image (similar to tab2 logic)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Your existing processing logic here
                            pass
                    
                    st.image(image, caption="Captured Gesture", use_column_width=True)
        
        with col2:
            st.subheader("📊 Current Status")
            
            # Real-time prediction display
            if 'ctx' in locals() and ctx.video_transformer:
                prediction_placeholder = st.empty()
                
                # Update prediction display
                if hasattr(ctx.video_transformer, 'current_prediction'):
                    current_pred = ctx.video_transformer.current_prediction
                    if "No gesture detected" not in current_pred and "Processing" not in current_pred:
                        prediction_placeholder.success(f"**Detected:** {current_pred}")
                    else:
                        prediction_placeholder.info(f"**Status:** {current_pred}")
                else:
                    prediction_placeholder.info("**Status:** Initializing...")
            
            # Performance tips
            st.subheader("💡 Tips for Better Results")
            st.write("""
            **Optimize Performance:**
            • Good lighting on hands
            • Plain background
            • Clear hand positioning
            • Stable hand movements
            • Proper camera distance
            
            **Troubleshooting:**
            • Refresh if video freezes
            • Check camera permissions
            • Try different confidence levels
            • Ensure stable internet connection
            • Use Chrome browser for best results
            """)
    
    # Keep your existing tab2 and tab3 code unchanged
    with tab2:
        st.subheader("Upload Image for Sign Detection")
        st.write("Upload an image containing sign language gestures for detection.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Your existing image processing code
            mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
            hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=detection_confidence)
            
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            predictions = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []
                    H, W, _ = frame.shape

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10

                    if model and len(data_aux) == 42:
                        try:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                            predictions.append(predicted_character)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_rgb, use_column_width=True)
                
                if predictions:
                    st.success(f"**Detected Gestures:** {', '.join(predictions)}")
                else:
                    st.warning("No hand gestures detected. Try adjusting the detection confidence.")
    
    with tab3:
        # Your existing About tab content
        st.subheader("About This Real-Time Application")
        st.write("""
        This enhanced Sign Language Recognition System provides **real-time video streaming** 
        and gesture detection using advanced web technologies.
        
        ## 🚀 Key Features
        
        **Real-Time Processing:**
        - Live webcam streaming via WebRTC
        - Instantaneous gesture recognition
        - Real-time prediction display
        - Adjustable detection parameters
        """)

if __name__ == "__main__":
    main()
