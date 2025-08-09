import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import base64
import io
import time
import threading
from queue import Queue

# Configure page
st.set_page_config(
    page_title="Real-Time Gesture Recognition",
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

# Labels dictionary
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

def process_frame_data(frame_data, model, mp_hands, mp_drawing, mp_drawing_styles):
    """Process base64 frame data from JavaScript"""
    try:
        # Decode base64 image
        header, encoded = frame_data.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize hands detector
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
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
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Predict
                if model and len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        predictions.append(predicted_character)
                        
                        # Draw on frame
                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10
                        x2 = int(max(x_) * W) + 10
                        y2 = int(max(y_) * H) + 10
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    except Exception as e:
                        pass

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Convert back to RGB
        processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return processed_rgb, predictions
        
    except Exception as e:
        st.error(f"Frame processing error: {str(e)}")
        return None, []

def main():
    st.title("🎥 Real-Time Gesture Recognition (Native Camera)")
    st.markdown("**Using browser's native camera API for real-time processing**")
    
    # Load model
    model = load_model()
    if not model:
        st.error("Cannot start without model")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
    
    # Initialize session state
    if 'live_predictions' not in st.session_state:
        st.session_state.live_predictions = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # Native camera component
        camera_html = """
        <div id="camera-container">
            <video id="video" width="640" height="480" autoplay muted playsinline></video>
            <canvas id="overlay" width="640" height="480" style="position: absolute; border: 2px solid #00ff00;"></canvas>
            <div id="controls" style="margin-top: 10px;">
                <button onclick="startCamera()" style="background: #4CAF50; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px;">Start Camera</button>
                <button onclick="toggleProcessing()" style="background: #2196F3; color: white; border: none; padding: 10px 20px; margin: 5px; border-radius: 5px;">Start Live Processing</button>
            </div>
            <div id="status" style="margin-top: 10px; font-weight: bold;">Camera Status: Ready</div>
            <div id="prediction" style="margin-top: 5px; font-size: 18px; color: #00ff00;">Prediction: None</div>
        </div>

        <script>
        let video = document.getElementById('video');
        let overlay = document.getElementById('overlay');
        let ctx = overlay.getContext('2d');
        let stream = null;
        let isProcessing = false;
        let processingInterval = null;

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 },
                        facingMode: 'user'
                    }
                });
                video.srcObject = stream;
                document.getElementById('status').innerText = 'Camera Status: Active';
            } catch (err) {
                console.error('Camera error:', err);
                document.getElementById('status').innerText = 'Camera Status: Error - ' + err.message;
            }
        }

        function captureAndProcess() {
            if (video.videoWidth === 0) return;
            
            // Create temporary canvas for capture
            let canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            let tempCtx = canvas.getContext('2d');
            
            // Draw current video frame
            tempCtx.drawImage(video, 0, 0, 640, 480);
            let imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send to Streamlit for processing
            fetch('/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frame_data: imageData
                })
            }).then(response => response.json())
              .then(data => {
                  if (data.predictions && data.predictions.length > 0) {
                      document.getElementById('prediction').innerText = 'Prediction: ' + data.predictions.join(', ');
                      document.getElementById('prediction').style.color = '#00ff00';
                  } else {
                      document.getElementById('prediction').innerText = 'Prediction: No gesture detected';
                      document.getElementById('prediction').style.color = '#ffaa00';
                  }
              }).catch(err => {
                  console.error('Processing error:', err);
              });
        }

        function toggleProcessing() {
            isProcessing = !isProcessing;
            
            if (isProcessing) {
                processingInterval = setInterval(captureAndProcess, 200); // 5 FPS processing
                document.getElementById('status').innerText = 'Camera Status: Live Processing Active';
            } else {
                clearInterval(processingInterval);
                document.getElementById('status').innerText = 'Camera Status: Active';
            }
        }

        // Auto-start camera
        window.addEventListener('load', function() {
            setTimeout(startCamera, 1000);
        });
        </script>

        <style>
        #camera-container {
            position: relative;
            display: inline-block;
            text-align: center;
        }
        #overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        </style>
        """
        
        st.components.v1.html(camera_html, height=600)
    
    with col2:
        st.subheader("📊 Live Status")
        
        # Live prediction display
        if st.session_state.live_predictions:
            latest_prediction = st.session_state.live_predictions[-1]
            st.success(f"**Latest:** {latest_prediction}")
        else:
            st.info("**Status:** Waiting for gestures...")
        
        # Statistics
        st.metric("Frames Processed", st.session_state.frame_count)
        
        # Recent predictions
        if len(st.session_state.live_predictions) > 1:
            st.subheader("🕒 Recent Detections")
            for i, pred in enumerate(st.session_state.live_predictions[-5:]):
                st.write(f"{i+1}. {pred}")
        
        # Instructions
        st.subheader("💡 Instructions")
        st.write("""
        **Steps:**
        1. Click "Start Camera"
        2. Allow camera permissions
        3. Click "Start Live Processing"
        4. Show gestures to camera
        
        **Tips:**
        • Good lighting helps
        • Keep hands steady
        • One gesture at a time
        • Stay in camera frame
        """)

if __name__ == "__main__":
    main()
