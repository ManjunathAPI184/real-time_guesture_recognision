import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import tempfile
import os

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

# Initialize MediaPipe
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
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
            if model and len(data_aux) == 42:  # Ensure we have the right number of features
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
    
    # Load model and initialize MediaPipe
    model = load_model()
    if model is None:
        return
    
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.1)
    
    # Update hands configuration
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=detection_confidence)
    
    # Display available gestures
    st.sidebar.subheader("Available Gestures")
    for key, value in labels_dict.items():
        st.sidebar.write(f"**{key}:** {value}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📷 Real-time Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.subheader("Real-time Sign Language Detection")
        st.write("Use your webcam to detect sign language gestures in real-time.")
        
        # Webcam input
        run_webcam = st.checkbox("Start Webcam")
        
        if run_webcam:
            # Create placeholder for video
            frame_placeholder = st.empty()
            prediction_placeholder = st.empty()
            
            # Initialize webcam
            cap = cv2.VideoCapture(1)
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera permissions.")
                return
            
            stop_button = st.button("Stop", key="stop_webcam")
            
            while run_webcam and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Process frame
                processed_frame = process_frame(frame, hands, mp_drawing, mp_hands, 
                                              mp_drawing_styles, model)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Check if stop button is pressed
                if st.session_state.get('stop_webcam', False):
                    break
            
            cap.release()
    
    with tab2:
        st.subheader("Upload Image for Sign Detection")
        st.write("Upload an image containing sign language gestures for detection.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Read and display original image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Process the image
            processed_frame = process_frame(frame, hands, mp_drawing, mp_hands, 
                                          mp_drawing_styles, model)
            
            # Convert back to RGB for display
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_rgb, use_column_width=True)
    
    with tab3:
        st.subheader("About This Application")
        st.write("""
        This Sign Language Recognition System uses computer vision and machine learning to detect and classify sign language gestures in real-time.
        
        **Features:**
        - Real-time webcam detection
        - Image upload for batch processing
        - Support for 20 different sign language gestures
        - Adjustable detection confidence
        
        **Technology Stack:**
        - **Streamlit:** Web application framework
        - **OpenCV:** Computer vision library
        - **MediaPipe:** Hand landmark detection
        - **Scikit-learn:** Machine learning model
        
        **How to Use:**
        1. Go to the "Real-time Detection" tab
        2. Click "Start Webcam" to begin detection
        3. Show sign language gestures to your camera
        4. The system will detect and display the predicted gesture
        
        **Supported Gestures:**
        """)
        
        # Display gestures in a nice format
        col1, col2 = st.columns(2)
        gestures_list = list(labels_dict.values())
        mid = len(gestures_list) // 2
        
        with col1:
            for gesture in gestures_list[:mid]:
                st.write(f"• {gesture}")
        
        with col2:
            for gesture in gestures_list[mid:]:
                st.write(f"• {gesture}")

if __name__ == "__main__":
    main()

