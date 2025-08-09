import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        with open('./model.p', 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    except FileNotFoundError:
        st.error("❌ Model file 'model.p' not found. Please ensure the model file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Initialize MediaPipe with caching
@st.cache_resource
def init_mediapipe():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp_hands, mp_drawing, mp_drawing_styles

# Enhanced labels dictionary with categories
labels_dict = {
    0: 'I', 1: 'YOU', 2: 'LOVE', 3: 'HATE', 4: 'OK', 5: 'NOT OK',
    6: 'WIN', 7: 'SUPER', 8: 'HELP', 9: 'STOP', 10: 'COME', 11: 'GO',
    12: 'THANK YOU', 13: 'SORRY', 14: 'YES', 15: 'NO', 16: 'PLEASE',
    17: 'GOOD MORNING', 18: 'GOODBYE', 19: 'WELCOME'
}

# Categorize gestures for better organization
gesture_categories = {
    "Personal": ['I', 'YOU'],
    "Emotions": ['LOVE', 'HATE'],
    "Status": ['OK', 'NOT OK', 'WIN', 'SUPER'],
    "Actions": ['HELP', 'STOP', 'COME', 'GO'],
    "Courtesy": ['THANK YOU', 'SORRY', 'PLEASE', 'WELCOME'],
    "Responses": ['YES', 'NO'],
    "Greetings": ['GOOD MORNING', 'GOODBYE']
}

# Initialize session state for analytics
def init_session_state():
    if 'gesture_history' not in st.session_state:
        st.session_state.gesture_history = []
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'total_predictions': 0,
            'successful_detections': 0,
            'session_start': datetime.now(),
            'gesture_counts': {}
        }
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []

# Enhanced gesture processing function
def process_gesture_image(image, model, mp_hands, mp_drawing, mp_drawing_styles, confidence_threshold=0.3):
    """Enhanced image processing with detailed analytics"""
    start_time = time.time()
    
    # Convert PIL to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_frame = frame.copy()
    
    # Initialize hands detector
    hands = mp_hands.Hands(
        static_image_mode=True, 
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=0.5
    )
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predictions = []
    confidence_scores = []
    hand_count = 0
    
    if results.multi_hand_landmarks:
        hand_count = len(results.multi_hand_landmarks)
        
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

            # Make prediction
            if model and len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    confidence_score = prediction_proba.max()
                    
                    predictions.append(predicted_character)
                    confidence_scores.append(confidence_score)
                    
                    # Draw bounding box and prediction
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    
                    # Color code based on confidence
                    if confidence_score > 0.8:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence_score > 0.6:
                        color = (255, 255, 0)  # Yellow for medium confidence
                    else:
                        color = (255, 0, 0)  # Red for low confidence
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    
                    # Add prediction text with confidence
                    label = f"{predicted_character} ({confidence_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

            # Draw hand landmarks with enhanced styling
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    processing_time = time.time() - start_time
    
    # Convert back to RGB for display
    processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return {
        'processed_image': processed_rgb,
        'original_image': cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
        'predictions': predictions,
        'confidence_scores': confidence_scores,
        'hand_count': hand_count,
        'processing_time': processing_time
    }

# Update session statistics
def update_stats(predictions, confidence_scores, processing_time):
    st.session_state.session_stats['total_predictions'] += 1
    st.session_state.processing_times.append(processing_time)
    
    if predictions:
        st.session_state.session_stats['successful_detections'] += 1
        for pred in predictions:
            if pred in st.session_state.session_stats['gesture_counts']:
                st.session_state.session_stats['gesture_counts'][pred] += 1
            else:
                st.session_state.session_stats['gesture_counts'][pred] = 1
        
        # Add to history (keep last 20)
        st.session_state.gesture_history.extend(predictions)
        if len(st.session_state.gesture_history) > 20:
            st.session_state.gesture_history = st.session_state.gesture_history[-20:]

# Main application
def main():
    init_session_state()
    
    # Header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🤟 Advanced Sign Language Recognition System</h1>
        <p>AI-Powered Gesture Detection with Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Cannot proceed without model. Please check your model file.")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
    
    # Sidebar configuration with enhanced controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Detection settings
        detection_confidence = st.slider(
            "Detection Confidence", 
            0.1, 1.0, 0.3, 0.05,
            help="Higher values require more confident detections"
        )
        
        st.divider()
        
        # Session Statistics
        st.subheader("📊 Session Analytics")
        
        stats = st.session_state.session_stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Attempts", stats['total_predictions'])
            st.metric("Successful", stats['successful_detections'])
        
        with col2:
            success_rate = (stats['successful_detections'] / max(stats['total_predictions'], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
            
            if st.session_state.processing_times:
                avg_time = np.mean(st.session_state.processing_times)
                st.metric("Avg Time", f"{avg_time:.2f}s")
        
        # Recent gestures history
        if st.session_state.gesture_history:
            st.subheader("🕒 Recent Gestures")
            recent_gestures = st.session_state.gesture_history[-5:]
            for i, gesture in enumerate(reversed(recent_gestures)):
                st.write(f"{i+1}. **{gesture}**")
        
        st.divider()
        
        # Gesture reference organized by category
        st.subheader("🔤 Gesture Reference")
        for category, gestures in gesture_categories.items():
            with st.expander(f"{category} ({len(gestures)})"):
                for gesture in gestures:
                    # Show count if gesture has been detected
                    count = stats['gesture_counts'].get(gesture, 0)
                    if count > 0:
                        st.write(f"• **{gesture}** _{count}x_")
                    else:
                        st.write(f"• {gesture}")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.gesture_history = []
            st.session_state.session_stats = {
                'total_predictions': 0,
                'successful_detections': 0,
                'session_start': datetime.now(),
                'gesture_counts': {}
            }
            st.session_state.processing_times = []
            st.rerun()
    
    # Main content with enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Camera Detection", 
        "📁 Upload Image", 
        "📈 Analytics Dashboard",
        "ℹ️ About"
    ])
    
    with tab1:
        st.subheader("📸 Camera Gesture Detection")
        st.info("📱 Capture clear hand gestures with good lighting for best results")
        
        # Camera input with enhanced feedback
        camera_photo = st.camera_input("📷 Capture your gesture")
        
        if camera_photo is not None:
            with st.spinner("🔄 Processing gesture..."):
                image = Image.open(camera_photo)
                
                # Process the image
                result = process_gesture_image(
                    image, model, mp_hands, mp_drawing, mp_drawing_styles, detection_confidence
                )
                
                # Update statistics
                update_stats(result['predictions'], result['confidence_scores'], result['processing_time'])
                
                # Display results in enhanced layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📸 Original Capture")
                    st.image(result['original_image'], use_column_width=True)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(result['processed_image'], use_column_width=True)
                
                # Results summary with enhanced styling
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hands Detected", result['hand_count'])
                with col2:
                    st.metric("Gestures Found", len(result['predictions']))
                with col3:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                with col4:
                    if result['confidence_scores']:
                        avg_conf = np.mean(result['confidence_scores'])
                        st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                # Detailed predictions
                if result['predictions']:
                    st.markdown("### 🎯 Detected Gestures")
                    for i, (pred, conf) in enumerate(zip(result['predictions'], result['confidence_scores'])):
                        if conf > 0.8:
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>Gesture {i+1}:</strong> {pred} 
                                <span style="color: green;">(High Confidence: {conf:.2f})</span>
                            </div>
                            """, unsafe_allow_html=True)
                        elif conf > 0.6:
                            st.markdown(f"""
                            <div class="warning-box">
                                <strong>Gesture {i+1}:</strong> {pred} 
                                <span style="color: orange;">(Medium Confidence: {conf:.2f})</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"**Gesture {i+1}:** {pred} (Low Confidence: {conf:.2f})")
                else:
                    st.warning("🤔 No gestures detected. Try:")
                    st.write("• Better lighting on your hands")
                    st.write("• Plain background")
                    st.write("• Clear hand positioning")
                    st.write("• Lower detection confidence")
    
    with tab2:
        st.subheader("📁 Upload Image Detection")
        st.info("📤 Upload images containing sign language gestures")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            with st.spinner("🔄 Processing uploaded image..."):
                image = Image.open(uploaded_file)
                
                # Process the image
                result = process_gesture_image(
                    image, model, mp_hands, mp_drawing, mp_drawing_styles, detection_confidence
                )
                
                # Update statistics
                update_stats(result['predictions'], result['confidence_scores'], result['processing_time'])
                
                # Display results (same layout as camera detection)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📤 Uploaded Image")
                    st.image(result['original_image'], use_column_width=True)
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(result['processed_image'], use_column_width=True)
                
                # Results summary (same as camera detection)
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Hands Detected", result['hand_count'])
                with col2:
                    st.metric("Gestures Found", len(result['predictions']))
                with col3:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                with col4:
                    if result['confidence_scores']:
                        avg_conf = np.mean(result['confidence_scores'])
                        st.metric("Avg Confidence", f"{avg_conf:.2f}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                # Detailed predictions
                if result['predictions']:
                    st.success(f"✅ **Detected Gestures:** {', '.join(result['predictions'])}")
                else:
                    st.warning("🤔 No gestures detected in the uploaded image.")
    
    with tab3:
        st.subheader("📈 Analytics Dashboard")
        
        if st.session_state.session_stats['total_predictions'] > 0:
            stats = st.session_state.session_stats
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Attempts", stats['total_predictions'])
            with col2:
                st.metric("Successful Detections", stats['successful_detections'])
            with col3:
                success_rate = (stats['successful_detections'] / stats['total_predictions']) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            with col4:
                session_duration = datetime.now() - stats['session_start']
                st.metric("Session Duration", f"{session_duration.seconds // 60}m")
            
            # Performance metrics
            if st.session_state.processing_times:
                st.subheader("⚡ Performance Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    avg_time = np.mean(st.session_state.processing_times)
                    st.metric("Average Processing Time", f"{avg_time:.2f}s")
                    
                with col2:
                    min_time = np.min(st.session_state.processing_times)
                    max_time = np.max(st.session_state.processing_times)
                    st.metric("Time Range", f"{min_time:.2f}s - {max_time:.2f}s")
            
            # Gesture frequency analysis
            if stats['gesture_counts']:
                st.subheader("🔤 Most Detected Gestures")
                
                # Sort gestures by frequency
                sorted_gestures = sorted(
                    stats['gesture_counts'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                # Display top 5 gestures
                for i, (gesture, count) in enumerate(sorted_gestures[:5]):
                    percentage = (count / stats['successful_detections']) * 100
                    st.write(f"{i+1}. **{gesture}**: {count} times ({percentage:.1f}%)")
            
        else:
            st.info("📊 No data available yet. Start detecting gestures to see analytics!")
    
    with tab4:
        st.subheader("ℹ️ About This Application")
        
        st.markdown("""
        ### 🚀 Advanced Sign Language Recognition System
        
        This application demonstrates state-of-the-art computer vision and machine learning 
        technologies for real-time sign language gesture recognition.
        
        #### 🔧 Technical Features
        
        **Core Technologies:**
        - **MediaPipe**: Google's hand landmark detection
        - **OpenCV**: Computer vision processing
        - **Scikit-learn**: Machine learning classification
        - **Streamlit**: Interactive web interface
        
        **Advanced Features:**
        - Real-time confidence scoring
        - Session analytics and tracking
        - Color-coded confidence indicators
        - Gesture frequency analysis
        - Performance metrics monitoring
        
        #### 📊 Recognition Capabilities
        
        **Supported Gestures (20 total):**
        """)
        
        # Display gesture categories in a nice format
        for category, gestures in gesture_categories.items():
            st.write(f"**{category}:** {', '.join(gestures)}")
        
        st.markdown("""
        
        #### 🎯 Performance Specifications
        
        - **Processing Speed**: < 2 seconds per image
        - **Accuracy**: Depends on image quality and lighting
        - **Hand Detection**: Up to 2 hands simultaneously
        - **Confidence Scoring**: Probabilistic output (0.0 - 1.0)
        
        #### 💡 Usage Tips
        
        **For Best Results:**
        - Use good lighting on your hands
        - Keep hands clearly visible
        - Use plain backgrounds
        - Make clear, distinct gestures
        - Adjust confidence threshold as needed
        
        **Troubleshooting:**
        - Lower confidence for difficult gestures
        - Ensure entire hand is in frame
        - Try different hand positions
        - Check for adequate lighting
        """)
        
        # Technical specifications
        st.subheader("🔍 Technical Specifications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Details:**")
            st.write("• Hand landmarks: 21 key points")
            st.write("• Feature vector: 42 dimensions")
            st.write("• Normalization: Min-max scaling")
            st.write("• Classification: ML algorithm")
        
        with col2:
            st.write("**Performance:**")
            st.write("• Image formats: JPG, PNG, JPEG")
            st.write("• Processing: Real-time")
            st.write("• Multi-hand: Up to 2 hands")
            st.write("• Confidence: Probabilistic scoring")

if __name__ == "__main__":
    main()
