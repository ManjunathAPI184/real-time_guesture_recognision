import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Configure Streamlit page
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_dict = pickle.load(open('./model.p', 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        st.error("Model file 'model.p' not found. Please ensure the model file is in the same directory.")
        return None

# Initialize MediaPipe for video mode
@st.cache_resource
def init_mediapipe(detection_conf=0.3):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,  # ✅ Video mode for real-time
        max_num_hands=2,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=0.3
    )
    return mp_hands, mp_drawing, mp_drawing_styles, hands

# Labels dictionary
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

def process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model):
    """Process a single frame and return the annotated frame with predictions"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            H, W, _ = frame.shape

            # Extract landmark coordinates
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            # Normalize coordinates
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Make prediction
            if model and len(data_aux) == 42:  # Ensure correct feature count
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return frame

def main():
    st.title("🤟 Sign Language Recognition System")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Sidebar config
    st.sidebar.header("Configuration")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.1)
    
    # Init mediapipe
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe(detection_confidence)
    
    # Sidebar gestures
    st.sidebar.subheader("Available Gestures")
    for key, value in labels_dict.items():
        st.sidebar.write(f"**{key}:** {value}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📷 Real-time Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.subheader("Real-time Sign Language Detection")
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check permissions.")
                return
            
            frame_placeholder = st.empty()
            stop_button = st.button("Stop Webcam")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                frame = cv2.flip(frame, 1)
                processed_frame = process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model)
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                if stop_button:
                    break
            
            cap.release()

    with tab2:
        st.subheader("Upload Image for Sign Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_frame = process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_rgb, use_column_width=True)

    with tab3:
        st.subheader("About This Application")
        st.write("""
        This system uses OpenCV, MediaPipe, and a trained ML model to detect sign language gestures in real-time.
        - **Real-time webcam detection**
        - **Image upload for batch processing**
        - Supports 20 different gestures
        - Adjustable detection confidence
        """)

if __name__ == "__main__":
    main()
