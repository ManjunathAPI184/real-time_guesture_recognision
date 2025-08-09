import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

# Configure page
st.set_page_config(
    page_title="Real-Time Sign Language Recognition",
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
    mp_hands = mp.solutions.hands  # Fixed the typo here
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

def process_frame(image, model, mp_hands, mp_drawing, mp_drawing_styles, hands):
    """Process image exactly like your original code"""
    # Convert PIL to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Your original processing logic
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

            # Make prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                predictions.append(predicted_character)

                # Draw exactly like your original code
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            except Exception as e:
                st.write(f"Prediction error: {str(e)}")

            # Draw landmarks exactly like your original code
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Convert back to RGB for display
    processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return processed_rgb, predictions

def main():
    st.title("🤟 Real-Time Sign Language Recognition")
    st.markdown("**Converted from your working local OpenCV code**")
    
    # Load model
    model = load_model()
    if not model:
        st.error("Cannot start without model")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles, hands = init_mediapipe()
    
    # Initialize session state for continuous processing
    if 'auto_process' not in st.session_state:
        st.session_state.auto_process = False
    if 'frame_counter' not in st.session_state:
        st.session_state.frame_counter = 0
    if 'last_predictions' not in st.session_state:
        st.session_state.last_predictions = []
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Auto-Processing"):
            st.session_state.auto_process = True
            
    with col2:
        if st.button("⏹️ Stop Auto-Processing"):
            st.session_state.auto_process = False
            
    with col3:
        st.write(f"**Status:** {'🟢 Running' if st.session_state.auto_process else '🔴 Stopped'}")
    
    # Main processing area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Auto-updating camera input for quasi-real-time
        if st.session_state.auto_process:
            # Create unique key to force refresh
            camera_key = f"camera_{int(time.time())}"
            auto_refresh_placeholder = st.empty()
        else:
            camera_key = "camera_static"
            
        camera_photo = st.camera_input(
            "📷 Capture gesture", 
            key=camera_key
        )
        
        if camera_photo is not None:
            st.session_state.frame_counter += 1
            
            # Process the image
            image = Image.open(camera_photo)
            processed_image, predictions = process_frame(
                image, model, mp_hands, mp_drawing, mp_drawing_styles, hands
            )
            
            # Update predictions history
            if predictions:
                st.session_state.last_predictions.extend(predictions)
                if len(st.session_state.last_predictions) > 10:
                    st.session_state.last_predictions = st.session_state.last_predictions[-10:]
            
            # Display processed image
            st.image(processed_image, caption=f"Processed Frame #{st.session_state.frame_counter}", 
                    use_column_width=True)
            
            # Show predictions
            if predictions:
                st.success(f"**🎯 Detected:** {', '.join(predictions)}")
    
    with col2:
        st.subheader("📊 Results")
        
        # Current status
        if st.session_state.last_predictions:
            st.success(f"**Latest:** {st.session_state.last_predictions[-1]}")
        else:
            st.info("**Status:** Ready for gestures")
        
        # Statistics
        st.metric("Frames Processed", st.session_state.frame_counter)
        
        # Recent predictions
        if st.session_state.last_predictions:
            st.subheader("🕒 Recent Detections")
            recent = st.session_state.last_predictions[-5:]
            for i, pred in enumerate(reversed(recent)):
                st.write(f"{i+1}. **{pred}**")
        
        # Clear history
        if st.button("🗑️ Clear History"):
            st.session_state.last_predictions = []
            st.session_state.frame_counter = 0
            st.rerun()
        
        # Instructions
        st.subheader("💡 How to Use")
        st.write("""
        **Steps:**
        1. Click **"Start Auto-Processing"**
        2. Allow camera permissions
        3. Show gesture to camera
        4. Camera auto-captures every few seconds
        
        **Tips:**
        • Good lighting on hands
        • Clear background
        • Hold gesture steady
        • One hand works better
        """)
    
    # Auto-refresh logic for continuous processing
    if st.session_state.auto_process:
        time.sleep(2)  # Wait 2 seconds between captures
        st.rerun()

if __name__ == "__main__":
    main()
