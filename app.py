import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Configure page
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

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

# Labels
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

def main():
    st.title("🤟 Sign Language Recognition System")
    st.success("✅ App is running successfully!")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        return
        
    mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
    
    st.info("📱 This is a simplified version without WebRTC to ensure stability")
    
    # Camera input method
    st.subheader("📸 Camera Gesture Detection")
    camera_photo = st.camera_input("Capture your gesture")
    
    if camera_photo is not None:
        # Process image
        image = Image.open(camera_photo)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize hands
        hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        predictions = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                H, W, _ = frame.shape

                # Extract landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Predict
                if len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        predictions.append(predicted_character)
                        
                        # Draw on frame
                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        x2 = int(max(x_) * W) + 10
                        y2 = int(max(y_) * H) + 10
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Processed")
            processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, use_column_width=True)
        
        if predictions:
            st.success(f"**Detected:** {', '.join(predictions)}")
        else:
            st.info("No gestures detected")
    
    # File upload
    st.subheader("📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Same processing logic as camera input
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

if __name__ == "__main__":
    main()
