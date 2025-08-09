import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Configure page
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="wide")

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
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    return mp_hands, mp_drawing, mp_drawing_styles, hands

# Gesture labels
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

# Frame processing
def process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []
            H, W, _ = frame.shape

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            if model and len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    return frame

# Main App
def main():
    st.title("🤟 Real-time Sign Language Recognition")
    st.markdown("Detect and translate sign language gestures directly in your browser using AI.")

    model = load_model()
    if model is None:
        return

    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()

    tab1, tab2 = st.tabs(["📷 Real-time Detection", "ℹ️ About"])

    with tab1:
        st.subheader("Use your webcam to detect gestures")
        img_file = st.camera_input("Show a sign gesture to your camera")

        if img_file is not None:
            image = Image.open(img_file)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_frame = process_frame(frame, hands, mp_drawing, mp_hands, mp_drawing_styles, model)
            processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(processed_rgb, channels="RGB", use_column_width=True)

    with tab2:
        st.subheader("About This Project")
        st.write("""
        This AI-powered Sign Language Recognition System uses computer vision and 
        machine learning to detect 20 different gestures in real-time.  

        **Features:**
        - Browser-based real-time detection (no software install needed)
        - Recognizes 20 gestures including greetings, commands, and basic responses
        - Powered by MediaPipe for hand tracking and Scikit-learn for classification  

        **Tech Stack:**
        - **Streamlit**: Web interface
        - **OpenCV**: Frame processing
        - **MediaPipe**: Landmark detection
        - **Scikit-learn**: Trained ML model
        - **NumPy/PIL**: Data processing  

        **Project Goal:**  
        To bridge communication gaps by enabling instant, accessible sign language translation.
        """)

if __name__ == "__main__":
    main()
