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
    
    predictions = []
    
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
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    predictions.append(predicted_character)
                    
                    # Draw bounding box and text
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return frame, predictions

def main():
    st.title("🤟 Sign Language Recognition System")
    st.markdown("---")
    
    # Add deployment info
    st.info("📱 **Note**: This version uses camera input for gesture detection. For local webcam streaming, run the app locally.")
    
    # Load model and initialize MediaPipe
    model = load_model()
    if model is None:
        return
    
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.3, 0.1)
    
    # Update hands configuration based on slider
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=detection_confidence)
    
    # Display available gestures
    st.sidebar.subheader("Available Gestures")
    for key, value in labels_dict.items():
        st.sidebar.write(f"**{key}:** {value}")
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["📷 Camera Detection", "📁 Upload Image", "ℹ️ About"])
    
    with tab1:
        st.subheader("Camera Sign Language Detection")
        st.write("Use your device camera to capture and detect sign language gestures.")
        
        # Create columns for different input methods
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Take Photo")
            camera_photo = st.camera_input("Capture your gesture")
            
            if camera_photo is not None:
                # Process camera photo
                image = Image.open(camera_photo)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process the captured image
                processed_frame, predictions = process_frame(frame, hands, mp_drawing, mp_hands, 
                                              mp_drawing_styles, model)
                
                # Display results
                st.subheader("📊 Detection Results")
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                st.image(processed_rgb, use_column_width=True)
                
                # Show predictions
                if predictions:
                    st.success(f"**Detected Gestures:** {', '.join(predictions)}")
                else:
                    st.warning("No hand gestures detected. Try adjusting the detection confidence or ensure your hand is clearly visible.")
        
        with col2:
            st.subheader("📋 Instructions")
            st.write("""
            **How to get the best results:**
            
            1. **Good Lighting**: Ensure adequate lighting on your hands
            2. **Clear Background**: Use a plain background behind your hands
            3. **Hand Position**: Keep your hands clearly visible in the frame
            4. **Single Gesture**: Make one clear gesture at a time
            5. **Distance**: Keep appropriate distance from camera
            
            **Tips:**
            - Adjust detection confidence using the sidebar slider
            - Try different hand positions if gesture isn't recognized
            - Make sure your entire hand is visible in the camera frame
            """)
            
            # Add a quick reference of common gestures
            st.subheader("🔤 Quick Reference")
            common_gestures = ['I', 'YOU', 'LOVE', 'OK', 'YES', 'NO', 'THANK YOU', 'HELP']
            for gesture in common_gestures:
                st.write(f"• **{gesture}**")
    
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
            processed_frame, predictions = process_frame(frame, hands, mp_drawing, mp_hands, 
                                          mp_drawing_styles, model)
            
            # Convert back to RGB for display
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_rgb, use_column_width=True)
                
                # Display predictions
                if predictions:
                    st.success(f"**Detected Gestures:** {', '.join(predictions)}")
                else:
                    st.warning("No hand gestures detected in the uploaded image.")
    
    with tab3:
        st.subheader("About This Application")
        st.write("""
        This Sign Language Recognition System uses computer vision and machine learning to detect and classify sign language gestures.
        
        **Features:**
        - 📷 Real-time camera capture and detection
        - 📁 Image upload for batch processing  
        - 🎯 Support for 20 different sign language gestures
        - ⚙️ Adjustable detection confidence
        - 🌐 Web-based interface accessible from any device
        
        **Technology Stack:**
        - **Streamlit:** Web application framework
        - **OpenCV:** Computer vision library for image processing
        - **MediaPipe:** Google's hand landmark detection
        - **Scikit-learn:** Machine learning model for gesture classification
        - **NumPy:** Numerical computing library
        
        **How to Use:**
        1. Go to the "Camera Detection" tab
        2. Click "Capture your gesture" to take a photo
        3. Make a clear sign language gesture in front of your camera
        4. The system will process and display the predicted gesture
        5. Adjust detection confidence if needed using the sidebar slider
        
        **Supported Gestures:**
        """)
        
        # Display gestures in a nice format
        col1, col2 = st.columns(2)
        gestures_list = list(labels_dict.values())
        mid = len(gestures_list) // 2
        
        with col1:
            st.write("**Basic Gestures:**")
            for gesture in gestures_list[:mid]:
                st.write(f"• {gesture}")
        
        with col2:
            st.write("**Advanced Gestures:**")
            for gesture in gestures_list[mid:]:
                st.write(f"• {gesture}")
        
        st.markdown("---")
        st.subheader("🔧 Technical Details")
        st.write("""
        **Model Architecture:**
        - Hand landmark detection using MediaPipe (21 key points per hand)
        - Feature extraction through coordinate normalization
        - Classification using trained scikit-learn model
        - Real-time processing with optimized performance
        
        **Performance:**
        - Processing time: < 1 second per image
        - Accuracy: Depends on image quality and gesture clarity
        - Supported formats: JPG, JPEG, PNG
        """)

if __name__ == "__main__":
    main()
