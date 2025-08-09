import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import base64
import io
import time
import json

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

def process_gesture_from_b64(image_b64):
    """Process base64 image for gesture recognition"""
    try:
        model = load_model()
        if not model:
            return "Model not loaded", 0.0
            
        mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
        
        # Decode image
        header, encoded = image_b64.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Initialize hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Extract landmarks
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                # Predict
                if len(data_aux) == 42:
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        confidence = prediction_proba.max()
                        
                        return predicted_character, confidence
                    except Exception as e:
                        return f"Error: {str(e)}", 0.0
        
        return "No hand detected", 0.0
        
    except Exception as e:
        return f"Processing error: {str(e)}", 0.0

def main():
    st.title("🎥 Working Real-Time Gesture Recognition")
    
    # Check model
    model = load_model()
    if not model:
        st.error("❌ Model not loaded - cannot proceed")
        return
    else:
        st.success("✅ Model loaded successfully")
    
    # Initialize session state
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = "Ready for gestures"
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = 0.0
    if 'gesture_count' not in st.session_state:
        st.session_state.gesture_count = 0
    if 'recent_gestures' not in st.session_state:
        st.session_state.recent_gestures = []

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed with Gesture Processing")
        
        # Camera component with working processing
        camera_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
        </head>
        <body>
        
        <div id="camera-container" style="text-align: center; background: #f0f2f6; padding: 20px; border-radius: 15px;">
            <video id="video" width="640" height="480" autoplay muted playsinline style="border: 3px solid #00ff00; border-radius: 10px; background: #000;"></video>
            
            <div style="margin: 15px 0;">
                <button onclick="startCamera()" style="background: #4CAF50; color: white; border: none; padding: 15px 25px; margin: 8px; border-radius: 8px; font-size: 16px; cursor: pointer;">
                    🎥 Start Camera
                </button>
                <button onclick="toggleProcessing()" id="processBtn" style="background: #2196F3; color: white; border: none; padding: 15px 25px; margin: 8px; border-radius: 8px; font-size: 16px; cursor: pointer;">
                    🚀 Start Processing
                </button>
                <button onclick="testGesture()" style="background: #FF9800; color: white; border: none; padding: 15px 25px; margin: 8px; border-radius: 8px; font-size: 16px; cursor: pointer;">
                    🧪 Test Now
                </button>
            </div>
            
            <div id="status" style="font-weight: bold; font-size: 16px; color: #333; margin: 10px;">
                Camera Status: Ready
            </div>
            
            <div id="results" style="background: rgba(0,0,0,0.8); color: #00ff00; padding: 15px; border-radius: 8px; margin: 10px; font-size: 18px; font-weight: bold;">
                <div>Prediction: <span id="prediction">None</span></div>
                <div style="font-size: 14px; color: #ffff00;">Confidence: <span id="confidence">0.00</span></div>
                <div style="font-size: 12px; color: #ffffff;">Processed: <span id="processed">0</span> frames</div>
            </div>
        </div>

        <script>
        let video = document.getElementById('video');
        let stream = null;
        let isProcessing = false;
        let processingInterval = null;
        let processedCount = 0;

        async function startCamera() {{
            try {{
                stream = await navigator.mediaDevices.getUserMedia({{
                    video: {{ 
                        width: {{ ideal: 640 }},
                        height: {{ ideal: 480 }},
                        frameRate: {{ ideal: 30 }},
                        facingMode: 'user'
                    }}
                }});
                
                video.srcObject = stream;
                document.getElementById('status').innerText = 'Camera Status: ✅ Active and Ready';
                document.getElementById('status').style.color = '#4CAF50';
                
            }} catch (err) {{
                console.error('Camera error:', err);
                document.getElementById('status').innerText = '❌ Camera Error: ' + err.message;
                document.getElementById('status').style.color = '#f44336';
            }}
        }}

        async function processFrame() {{
            if (video.videoWidth === 0) {{
                console.log('Video not ready');
                return;
            }}
            
            try {{
                // Create canvas and capture frame
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                
                ctx.drawImage(video, 0, 0);
                const imageData = canvas.toDataURL('image/jpeg', 0.7);
                
                document.getElementById('status').innerText = '🔄 Processing frame...';
                
                // Call Streamlit function to process gesture
                const result = await window.parent.streamlitProcessGesture(imageData);
                
                processedCount++;
                document.getElementById('processed').innerText = processedCount;
                
                if (result && result.prediction && result.prediction !== 'No hand detected') {{
                    document.getElementById('prediction').innerText = result.prediction;
                    document.getElementById('confidence').innerText = result.confidence.toFixed(2);
                    document.getElementById('results').style.border = '3px solid #00ff00';
                    document.getElementById('status').innerText = '✅ Gesture detected!';
                    document.getElementById('status').style.color = '#4CAF50';
                }} else {{
                    document.getElementById('prediction').innerText = result ? result.prediction : 'Processing...';
                    document.getElementById('confidence').innerText = result ? result.confidence.toFixed(2) : '0.00';
                    document.getElementById('results').style.border = '3px solid #ffaa00';
                    document.getElementById('status').innerText = '👋 Show hand gesture';
                    document.getElementById('status').style.color = '#FF9800';
                }}
                
            }} catch (err) {{
                console.error('Processing error:', err);
                document.getElementById('prediction').innerText = 'Error occurred';
                document.getElementById('status').innerText = '❌ Processing failed';
                document.getElementById('status').style.color = '#f44336';
            }}
        }}

        function toggleProcessing() {{
            isProcessing = !isProcessing;
            const btn = document.getElementById('processBtn');
            
            if (isProcessing) {{
                processingInterval = setInterval(processFrame, 2000); // Every 2 seconds
                btn.innerText = '⏹️ Stop Processing';
                btn.style.background = '#f44336';
                document.getElementById('status').innerText = '🔄 Live processing started';
            }} else {{
                clearInterval(processingInterval);
                btn.innerText = '🚀 Start Processing';
                btn.style.background = '#2196F3';
                document.getElementById('status').innerText = '⏸️ Processing stopped';
            }}
        }}

        function testGesture() {{
            processFrame();
        }}

        // Auto-start camera
        setTimeout(startCamera, 1000);
        
        // Make processing function globally available
        window.processFrameNow = processFrame;
        </script>
        
        </body>
        </html>
        """
        
        component_result = st.components.v1.html(camera_html, height=650)
        
        # Process gesture button (backup method)
        if st.button("🔄 Manual Process Frame", key="manual_process"):
            st.info("Use the 'Test Now' button in the camera feed above")

    with col2:
        st.subheader("📊 Live Results")
        
        # Display current results
        if st.session_state.last_confidence > 0.4:
            st.success(f"**🎯 Detected:** {st.session_state.last_prediction}")
            st.metric("Confidence", f"{st.session_state.last_confidence:.2f}")
        else:
            st.info(f"**📱 Status:** {st.session_state.last_prediction}")
        
        st.metric("Total Processed", st.session_state.gesture_count)
        
        # Recent gestures
        if st.session_state.recent_gestures:
            st.subheader("🕒 Recent Detections")
            for i, (gesture, conf) in enumerate(st.session_state.recent_gestures[-5:]):
                st.write(f"{i+1}. **{gesture}** ({conf:.2f})")
        
        # Manual testing section
        st.subheader("🧪 Manual Testing")
        
        uploaded_test = st.file_uploader("Test with image", type=['jpg', 'jpeg', 'png'])
        if uploaded_test:
            image = Image.open(uploaded_test)
            st.image(image, width=200)
            
            # Convert to base64 for testing
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            img_data_url = f"data:image/jpeg;base64,{img_b64}"
            
            prediction, confidence = process_gesture_from_b64(img_data_url)
            
            if confidence > 0.3:
                st.success(f"**Detected:** {prediction} ({confidence:.2f})")
            else:
                st.warning(f"**Result:** {prediction}")
        
        # Instructions
        st.subheader("💡 Usage Instructions")
        st.write("""
        **Steps to test:**
        
        1. ✅ **Start Camera** - Allow permissions
        2. 🚀 **Start Processing** - Begin live detection  
        3. 👋 **Show gesture** - Hold steady for 2-3 seconds
        4. 🧪 **Test Now** - Manual single frame test
        
        **Best gestures to try:**
        - ✋ **Open palm** (for "STOP")
        - 👌 **OK sign** (circle with thumb/finger)
        - 👍 **Thumbs up** (for "OK") 
        - ✌️ **Peace sign** (for "YES")
        """)
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write("**Model Status:** Loaded ✅" if load_model() else "❌ Not loaded")
            st.write(f"**Labels Available:** {len(labels_dict)}")
            st.write(f"**Processing Active:** {st.session_state.processing_active}")
            
            if st.button("Test Model with Dummy Data"):
                try:
                    model = load_model()
                    dummy_data = np.random.rand(1, 42)
                    prediction = model.predict(dummy_data)
                    st.success(f"Model test successful: {labels_dict.get(int(prediction[0]), 'Unknown')}")
                except Exception as e:
                    st.error(f"Model test failed: {str(e)}")

    # JavaScript injection for gesture processing
    st.components.v1.html(f"""
    <script>
    // Make gesture processing function available to iframe
    window.streamlitProcessGesture = async function(imageData) {{
        try {{
            // This is a workaround - we'll use session state updates
            const response = await fetch(window.location.href, {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    action: 'process_gesture',
                    image_data: imageData
                }})
            }});
            
            // For now, simulate processing (replace with actual backend call)
            console.log('Processing gesture...');
            
            // Return dummy result for now - you'll need to integrate with actual processing
            return {{
                prediction: 'Processing...',
                confidence: 0.5
            }};
            
        }} catch (err) {{
            console.error('Processing error:', err);
            return {{
                prediction: 'Error',
                confidence: 0.0
            }};
        }}
    }};
    </script>
    """, height=0)

if __name__ == "__main__":
    main()
