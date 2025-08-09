import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import base64
import io
import time

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

labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

def process_frame_for_gestures(image_data):
    """Process captured frame for gesture recognition"""
    try:
        # Load model and MediaPipe
        model = load_model()
        if not model:
            return None, []
            
        mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
        
        # Decode base64 image
        header, encoded = image_data.split(',', 1)
        img_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_data))
        
        # Convert to OpenCV
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize hands detector  
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower threshold for better detection
            min_tracking_confidence=0.3
        )
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        predictions = []
        confidence_scores = []
        
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

                # Normalize coordinates  
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Make prediction
                if len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        confidence = prediction_proba.max()
                        
                        # Only accept predictions with reasonable confidence
                        if confidence > 0.3:
                            predictions.append(predicted_character)
                            confidence_scores.append(confidence)
                            
                            # Draw bounding box
                            x1 = int(min(x_) * W) - 15
                            y1 = int(min(y_) * H) - 15  
                            x2 = int(max(x_) * W) + 15
                            y2 = int(max(y_) * H) + 15
                            
                            # Color based on confidence
                            if confidence > 0.7:
                                color = (0, 255, 0)  # Green
                            elif confidence > 0.5:
                                color = (255, 255, 0)  # Yellow  
                            else:
                                color = (255, 0, 0)  # Red
                                
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                            
                            # Add prediction text
                            label = f"{predicted_character} ({confidence:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                                       
                    except Exception as e:
                        st.write(f"Prediction error: {str(e)}")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return processed_b64, predictions, confidence_scores
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None, [], []

def main():
    st.title("🎥 Real-Time Gesture Recognition - Enhanced")
    st.markdown("**Large camera feed with working gesture detection**")
    
    # Check model
    model = load_model()
    if not model:
        st.error("Model not loaded!")
        return
    
    # Session state for results
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = "No gesture detected"
    if 'confidence_score' not in st.session_state:
        st.session_state.confidence_score = 0.0
    if 'gesture_history' not in st.session_state:
        st.session_state.gesture_history = []
        
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📹 Enhanced Camera Feed")
        
        # JavaScript component with gesture processing
        camera_component = st.components.v1.html(f"""
        <div id="camera-container">
            <video id="video" width="800" height="600" autoplay muted playsinline></video>
            <div id="controls" style="margin-top: 15px;">
                <button onclick="startCamera()" style="background: #4CAF50; color: white; border: none; padding: 15px 30px; margin: 10px; border-radius: 8px; font-size: 16px; cursor: pointer;">🎥 Start Camera</button>
                <button onclick="toggleProcessing()" style="background: #2196F3; color: white; border: none; padding: 15px 30px; margin: 10px; border-radius: 8px; font-size: 16px; cursor: pointer;">🚀 Start Processing</button>
            </div>
            <div id="status" style="margin-top: 15px; font-weight: bold; font-size: 18px; color: #333;">Camera Status: Ready</div>
            <div id="prediction" style="margin-top: 10px; font-size: 24px; color: #00ff00; font-weight: bold; background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; min-height: 60px;">
                <div>Prediction: <span id="pred-text">None</span></div>
                <div style="font-size: 16px; color: #ffff00;">Confidence: <span id="conf-text">0.00</span></div>
            </div>
        </div>

        <script>
        let video = document.getElementById('video');
        let stream = null;
        let isProcessing = false;
        let processingInterval = null;

        async function startCamera() {{
            try {{
                const constraints = {{
                    video: {{ 
                        width: {{ min: 800, ideal: 1280 }},
                        height: {{ min: 600, ideal: 720 }},
                        frameRate: {{ ideal: 30 }},
                        facingMode: 'user'
                    }}
                }};
                
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                
                video.onloadedmetadata = function() {{
                    document.getElementById('status').innerText = `Camera Status: Active (${{video.videoWidth}}x${{video.videoHeight}})`;
                }};
                
            }} catch (err) {{
                console.error('Camera error:', err);
                document.getElementById('status').innerText = 'Camera Status: Error - ' + err.message;
            }}
        }}

        function captureFrame() {{
            if (video.videoWidth === 0) return;
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to Streamlit backend via fetch
            fetch(window.location.href, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    type: 'frame_process',
                    frame_data: imageData
                }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.predictions && data.predictions.length > 0) {{
                    document.getElementById('pred-text').innerText = data.predictions.join(', ');
                    document.getElementById('conf-text').innerText = data.confidence_scores ? data.confidence_scores[0].toFixed(2) : '0.00';
                    document.getElementById('prediction').style.border = '3px solid #00ff00';
                }} else {{
                    document.getElementById('pred-text').innerText = 'No gesture detected';
                    document.getElementById('conf-text').innerText = '0.00';
                    document.getElementById('prediction').style.border = '3px solid #ffaa00';
                }}
            }})
            .catch(err => {{
                console.error('Processing error:', err);
                document.getElementById('pred-text').innerText = 'Processing error';
                document.getElementById('conf-text').innerText = '0.00';
            }});
        }}

        function toggleProcessing() {{
            isProcessing = !isProcessing;
            
            if (isProcessing) {{
                processingInterval = setInterval(captureFrame, 1000); // Process every 1 second
                document.getElementById('status').innerText = 'Camera Status: 🔄 Live Processing Active';
            }} else {{
                clearInterval(processingInterval);
                document.getElementById('status').innerText = 'Camera Status: Active';
            }}
        }}

        // Auto-start camera when loaded
        setTimeout(startCamera, 1000);
        </script>

        <style>
        #camera-container {{
            text-align: center;
            background: linear-gradient(145deg, #f0f2f6, #d1d9e6);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        #video {{
            border: 4px solid #00ff00;
            border-radius: 15px;
            background: #000;
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 20px rgba(0,255,0,0.3);
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        #prediction {{
            background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(0,0,0,0.6)) !important;
            border: 3px solid #00ff00;
            box-shadow: 0 0 20px rgba(0,255,0,0.3);
        }}
        </style>
        """, height=800)
    
    with col2:
        st.subheader("📊 Detection Results")
        
        # Current prediction display
        if st.session_state.current_prediction != "No gesture detected":
            st.success(f"**🎯 Current:** {st.session_state.current_prediction}")
            st.metric("Confidence", f"{st.session_state.confidence_score:.2f}")
        else:
            st.info("**📱 Status:** Ready for gestures...")
        
        # Gesture history
        if st.session_state.gesture_history:
            st.subheader("🕒 Recent Gestures")
            for i, gesture in enumerate(st.session_state.gesture_history[-5:]):
                st.write(f"{i+1}. **{gesture}**")
        
        # Tips for better detection
        st.subheader("💡 Detection Tips")
        st.write("""
        **For best results:**
        
        🔆 **Bright lighting** on hands
        
        🖐️ **Hold gesture steady** for 2-3 seconds
        
        📏 **Proper distance** from camera (arm's length)
        
        📱 **Clear background** behind hands
        
        ✋ **One hand** gestures work better
        
        🎯 **Center hands** in camera view
        """)
        
        # Supported gestures quick reference
        st.subheader("🔤 Quick Reference")
        common_gestures = ['I', 'YOU', 'OK', 'YES', 'NO', 'HELP', 'THANK YOU', 'STOP']
        for gesture in common_gestures:
            st.write(f"• **{gesture}**")
        
        if st.button("🗑️ Clear History"):
            st.session_state.gesture_history = []
            st.rerun()

if __name__ == "__main__":
    main()
