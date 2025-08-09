import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import pandas as pd
import os

# Configure page with professional styling
st.set_page_config(
    page_title="AI Sign Language Recognition System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .feature-card {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .gesture-card {
        background: white;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .gesture-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    .gesture-image {
        border-radius: 10px;
        border: 3px solid #f8f9fa;
        margin-bottom: 1rem;
    }
    
    .gesture-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .gesture-index {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #28a745; }
    .status-inactive { background-color: #dc3545; }
    .status-processing { background-color: #ffc107; }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff4757, #ffa502, #2ed573);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    
    .metric-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .guide-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .category-section {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load model with enhanced error handling
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

# Gesture categories for better organization
gesture_categories = {
    "👤 Personal": [0, 1],  # I, YOU
    "❤️ Emotions": [2, 3],  # LOVE, HATE
    "✅ Status": [4, 5, 6, 7],  # OK, NOT OK, WIN, SUPER
    "🚀 Actions": [8, 9, 10, 11],  # HELP, STOP, COME, GO
    "🙏 Courtesy": [12, 13, 16, 19],  # THANK YOU, SORRY, PLEASE, WELCOME
    "💬 Responses": [14, 15],  # YES, NO
    "👋 Greetings": [17, 18]  # GOOD MORNING, GOODBYE
}

# Gesture descriptions for better understanding
gesture_descriptions = {
    0: "Point to yourself with index finger",
    1: "Point towards the person you're talking to",
    2: "Cross hands over heart or make heart shape",
    3: "Firm fist or aggressive gesture",
    4: "Circle with thumb and index finger",
    5: "Shake head or wave hand dismissively",
    6: "Raise fist in victory pose",
    7: "Thumbs up with emphasis",
    8: "Open palm gesture, reaching out",
    9: "Open hand raised like stop sign",
    10: "Beckon with hand motion",
    11: "Point away or wave goodbye motion",
    12: "Hands together in prayer position",
    13: "Hand over heart with apologetic expression",
    14: "Nod or thumbs up gesture",
    15: "Shake head or wave hand",
    16: "Hands together in pleading position",
    17: "Wave hello with cheerful expression",
    18: "Wave farewell gesture",
    19: "Open arms in welcoming gesture"
}

# Function to check if gesture image exists
def check_gesture_image(gesture_index):
    """Check if gesture image exists in the images folder"""
    image_path = f"images/{gesture_index}.jpg"
    return os.path.exists(image_path)

# Function to load gesture image
def load_gesture_image(gesture_index):
    """Load gesture image from images folder"""
    image_path = f"images/{gesture_index}.jpg"
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # Create placeholder image if not found
            placeholder = Image.new('RGB', (200, 200), color='lightgray')
            return placeholder
    except Exception as e:
        # Return placeholder on error
        placeholder = Image.new('RGB', (200, 200), color='lightgray')
        return placeholder

# Initialize enhanced session state
def init_session_state():
    if 'auto_process' not in st.session_state:
        st.session_state.auto_process = False
    if 'gesture_history' not in st.session_state:
        st.session_state.gesture_history = []
    if 'confidence_history' not in st.session_state:
        st.session_state.confidence_history = []
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'total_detections': 0,
            'high_confidence_detections': 0,
            'session_start': datetime.now(),
            'gesture_counts': Counter(),
            'avg_confidence': 0.0,
            'processing_times': []
        }
    if 'live_predictions' not in st.session_state:
        st.session_state.live_predictions = []

# Enhanced gesture processing with confidence analysis
def process_gesture_image(image, model, mp_hands, mp_drawing, mp_drawing_styles, confidence_threshold=0.3):
    """Enhanced processing with detailed confidence analysis"""
    start_time = time.time()
    
    # Convert PIL to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_frame = frame.copy()
    
    # Initialize hands detector with optimized settings
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=0.5
    )
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    predictions = []
    confidence_scores = []
    hand_landmarks_data = []
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            data_aux = []
            x_ = []
            y_ = []
            H, W, _ = frame.shape

            # Extract landmark coordinates
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize coordinates
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Calculate hand bounding box
            x1 = max(0, int(min(x_) * W) - 20)
            y1 = max(0, int(min(y_) * H) - 20)
            x2 = min(W, int(max(x_) * W) + 20)
            y2 = min(H, int(max(y_) * H) + 20)

            # Make prediction with confidence analysis
            if model and len(data_aux) == 42:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    
                    predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                    confidence_score = prediction_proba.max()
                    
                    predictions.append(predicted_character)
                    confidence_scores.append(confidence_score)
                    
                    # Store detailed hand data for analysis
                    hand_landmarks_data.append({
                        'hand_id': hand_idx,
                        'prediction': predicted_character,
                        'confidence': confidence_score,
                        'bbox': (x1, y1, x2, y2),
                        'landmark_count': len(hand_landmarks.landmark)
                    })
                    
                    # Confidence-based color coding
                    if confidence_score >= 0.8:
                        color = (0, 255, 0)  # Green - High confidence
                        thickness = 4
                    elif confidence_score >= 0.6:
                        color = (255, 255, 0)  # Yellow - Medium confidence
                        thickness = 3
                    elif confidence_score >= 0.4:
                        color = (255, 165, 0)  # Orange - Low confidence
                        thickness = 2
                    else:
                        color = (255, 0, 0)  # Red - Very low confidence
                        thickness = 2
                    
                    # Enhanced bounding box with confidence indicator
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Multi-line text display
                    label_lines = [
                        f"{predicted_character}",
                        f"Conf: {confidence_score:.2f}",
                        f"Hand {hand_idx + 1}"
                    ]
                    
                    for i, line in enumerate(label_lines):
                        y_offset = y1 - 15 - (i * 25)
                        cv2.putText(frame, line, (x1, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    
                    # Confidence bar visualization
                    bar_width = x2 - x1
                    bar_height = 8
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + int(bar_width * confidence_score), y2 + 5 + bar_height), 
                                 color, -1)
                    cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 5 + bar_height), (255, 255, 255), 1)
                    
                except Exception as e:
                    st.error(f"Prediction error for hand {hand_idx + 1}: {str(e)}")

            # Enhanced landmark drawing
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    processing_time = time.time() - start_time
    processed_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return {
        'processed_image': processed_rgb,
        'original_image': cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB),
        'predictions': predictions,
        'confidence_scores': confidence_scores,
        'hand_count': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0,
        'processing_time': processing_time,
        'hand_landmarks_data': hand_landmarks_data
    }

# Update session statistics with enhanced metrics
def update_stats(predictions, confidence_scores, processing_time):
    stats = st.session_state.session_stats
    
    stats['total_detections'] += 1
    stats['processing_times'].append(processing_time)
    
    if predictions and confidence_scores:
        # Update gesture counts
        for pred in predictions:
            stats['gesture_counts'][pred] += 1
        
        # Update confidence statistics
        avg_conf = np.mean(confidence_scores)
        st.session_state.confidence_history.append(avg_conf)
        
        # High confidence detection tracking
        if avg_conf >= 0.7:
            stats['high_confidence_detections'] += 1
        
        # Update running average confidence
        all_confidences = st.session_state.confidence_history
        stats['avg_confidence'] = np.mean(all_confidences) if all_confidences else 0.0
        
        # Update gesture history
        timestamp = datetime.now().strftime("%H:%M:%S")
        for pred, conf in zip(predictions, confidence_scores):
            st.session_state.gesture_history.append({
                'gesture': pred,
                'confidence': conf,
                'timestamp': timestamp
            })
        
        # Keep only recent history
        if len(st.session_state.gesture_history) > 50:
            st.session_state.gesture_history = st.session_state.gesture_history[-50:]

# Create confidence visualization
def create_confidence_chart():
    if len(st.session_state.confidence_history) > 1:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=st.session_state.confidence_history[-20:],  # Last 20 predictions
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2')
        ))
        
        fig.update_layout(
            title="Recent Confidence Trends",
            yaxis_title="Confidence Score",
            xaxis_title="Recent Predictions",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.add_hline(y=0.7, line_dash="dash", line_color="green", 
                     annotation_text="High Confidence Threshold")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Confidence Threshold")
        
        return fig
    return None

# Create gesture frequency chart
def create_gesture_frequency_chart():
    if st.session_state.session_stats['gesture_counts']:
        gestures = list(st.session_state.session_stats['gesture_counts'].keys())
        counts = list(st.session_state.session_stats['gesture_counts'].values())
        
        fig = px.bar(
            x=counts, y=gestures,
            orientation='h',
            title="Most Detected Gestures",
            color=counts,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    return None

def main():
    init_session_state()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>🤟 AI-Powered Sign Language Recognition System</h1>
        <p>Professional-grade gesture detection with advanced confidence analysis and visual learning guide</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model with status indicator
    model = load_model()
    if model is None:
        st.error("❌ System cannot operate without the trained model. Please check your model file.")
        return
    
    mp_hands, mp_drawing, mp_drawing_styles = init_mediapipe()
    
    # Enhanced sidebar with professional styling
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        # System status indicator
        st.markdown(f"""
        <div class="feature-card">
            <h4>🔋 System Status</h4>
            <p><span class="status-indicator status-active"></span>Model: Loaded</p>
            <p><span class="status-indicator status-active"></span>MediaPipe: Ready</p>
            <p><span class="status-indicator {'status-active' if st.session_state.auto_process else 'status-inactive'}"></span>Processing: {'Active' if st.session_state.auto_process else 'Standby'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced configuration
        st.markdown("#### 🎛️ Detection Parameters")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            0.1, 1.0, 0.3, 0.05,
            help="Higher values require more confident detections"
        )
        
        processing_speed = st.selectbox(
            "Processing Speed",
            ["Fast (3s intervals)", "Balanced (2s intervals)", "Precise (1s intervals)"],
            index=1,
            help="Balance between speed and accuracy"
        )
        
        speed_map = {
            "Fast (3s intervals)": 3,
            "Balanced (2s intervals)": 2,
            "Precise (1s intervals)": 1
        }
        
        # Session statistics dashboard
        stats = st.session_state.session_stats
        st.markdown("#### 📊 Session Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Detections", stats['total_detections'])
            st.metric("High Confidence", stats['high_confidence_detections'])
        
        with col2:
            success_rate = (stats['high_confidence_detections'] / max(stats['total_detections'], 1)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
            st.metric("Avg Confidence", f"{stats['avg_confidence']:.2f}")
        
        # Session duration
        session_duration = datetime.now() - stats['session_start']
        st.metric("Session Duration", f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")
        
        # Control buttons
        st.markdown("#### 🎮 System Controls")
        if st.button("🗑️ Reset Session", type="secondary"):
            for key in ['gesture_history', 'confidence_history', 'session_stats']:
                if key in st.session_state:
                    del st.session_state[key]
            init_session_state()
            st.success("Session reset successfully!")
            st.rerun()

    # Main interface with enhanced tabs including gesture guide
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 Live Detection", 
        "📁 Image Upload", 
        "📖 Gesture Guide",
        "📈 Analytics Dashboard",
        "ℹ️ System Information"
    ])
    
    with tab1:
        st.markdown("### 📸 Real-Time Gesture Detection")
        
        # Control panel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 Start Auto-Processing", type="primary"):
                st.session_state.auto_process = True
                st.success("Auto-processing activated!")
                
        with col2:
            if st.button("⏹️ Stop Processing", type="secondary"):
                st.session_state.auto_process = False
                st.info("Processing stopped")
                
        with col3:
            status_emoji = "🟢" if st.session_state.auto_process else "🔴"
            st.markdown(f"**Status:** {status_emoji} {'Active' if st.session_state.auto_process else 'Inactive'}")
            
        with col4:
            if st.session_state.auto_process:
                st.markdown("**⏱️ Auto-refresh active**")
        
        # Main processing area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📹 Camera Feed")
            
            # Dynamic camera input
            camera_key = f"camera_{int(time.time())}" if st.session_state.auto_process else "camera_static"
            camera_photo = st.camera_input("📷 Capture gesture for analysis", key=camera_key)
            
            if camera_photo is not None:
                with st.spinner("🔄 Processing gesture with AI analysis..."):
                    image = Image.open(camera_photo)
                    
                    # Process with enhanced features
                    result = process_gesture_image(
                        image, model, mp_hands, mp_drawing, mp_drawing_styles, confidence_threshold
                    )
                    
                    # Update statistics
                    update_stats(result['predictions'], result['confidence_scores'], result['processing_time'])
                    
                    # Display processed image
                    st.image(result['processed_image'], 
                            caption=f"Processed in {result['processing_time']:.2f}s | {result['hand_count']} hands detected",
                            use_column_width=True)
        
        with col2:
            st.markdown("#### 📊 Live Results")
            
            if camera_photo is not None:
                # Display predictions with confidence
                if result['predictions']:
                    for i, (pred, conf) in enumerate(zip(result['predictions'], result['confidence_scores'])):
                        # Confidence-based styling
                        if conf >= 0.8:
                            st.success(f"**Hand {i+1}:** {pred} ({conf:.2f})")
                        elif conf >= 0.6:
                            st.warning(f"**Hand {i+1}:** {pred} ({conf:.2f})")
                        else:
                            st.info(f"**Hand {i+1}:** {pred} ({conf:.2f})")
                        
                        # Confidence bar visualization
                        st.progress(conf, text=f"Confidence: {conf:.1%}")
                else:
                    st.info("👋 No hands detected. Please show your hand clearly to the camera.")
                
                # Processing metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                with col2:
                    st.metric("Hands Detected", result['hand_count'])
            
            # Recent detections history
            if st.session_state.gesture_history:
                st.markdown("#### 🕒 Recent Detections")
                recent_detections = st.session_state.gesture_history[-5:]
                for detection in reversed(recent_detections):
                    conf_color = "🟢" if detection['confidence'] >= 0.7 else "🟡" if detection['confidence'] >= 0.5 else "🔴"
                    st.write(f"{conf_color} **{detection['gesture']}** ({detection['confidence']:.2f}) - {detection['timestamp']}")
        
        # Auto-refresh logic
        if st.session_state.auto_process:
            time.sleep(speed_map[processing_speed])
            st.rerun()
    
    with tab2:
        st.markdown("### 📁 Batch Image Processing")
        
        uploaded_file = st.file_uploader(
            "Upload image containing sign language gestures",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG. Best results with clear hand gestures and good lighting."
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📤 Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("#### 🎯 Analysis Results")
                
                with st.spinner("Analyzing uploaded image..."):
                    result = process_gesture_image(
                        image, model, mp_hands, mp_drawing, mp_drawing_styles, confidence_threshold
                    )
                    
                    update_stats(result['predictions'], result['confidence_scores'], result['processing_time'])
                
                st.image(result['processed_image'], use_column_width=True)
                
                # Detailed analysis
                if result['predictions']:
                    st.markdown("##### 📋 Detailed Analysis")
                    for i, data in enumerate(result['hand_landmarks_data']):
                        st.markdown(f"""
                        **Hand {data['hand_id'] + 1} Analysis:**
                        - **Gesture:** {data['prediction']}
                        - **Confidence:** {data['confidence']:.2f} ({data['confidence']:.1%})
                        - **Landmarks:** {data['landmark_count']} points detected
                        """)
                        
                        # Confidence assessment
                        if data['confidence'] >= 0.8:
                            st.success("High confidence detection ✅")
                        elif data['confidence'] >= 0.6:
                            st.warning("Medium confidence detection ⚠️")
                        else:
                            st.error("Low confidence detection ❌")
                else:
                    st.info("No gesture detected in the uploaded image. Try with better lighting or clearer hand positioning.")
    
    with tab3:
        st.markdown("""
        <div class="guide-header">
            <h2>📖 Complete Gesture Learning Guide</h2>
            <p>Visual reference for all 20 supported sign language gestures</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions for image setup
        st.info("""
        **📁 Image Setup Instructions:**
        1. Create a folder named `images` in your project root directory
        2. Add gesture images named `0.jpg`, `1.jpg`, `2.jpg`, ..., `19.jpg`
        3. Each image should clearly show the corresponding gesture
        4. Images will be automatically loaded and displayed below
        """)
        
        # Check if images folder exists
        if not os.path.exists("images"):
            st.warning("⚠️ Images folder not found. Please create an 'images' folder and add gesture photos.")
            st.markdown("**Expected folder structure:**")
            st.code("""
project-root/
├── app.py
├── model.p
├── images/
│   ├── 0.jpg    # Gesture for 'I'
│   ├── 1.jpg    # Gesture for 'YOU'
│   ├── 2.jpg    # Gesture for 'LOVE'
│   └── ...      # Continue for all 20 gestures
└── requirements.txt
            """)
        
        # Display gestures by category
        for category_name, gesture_indices in gesture_categories.items():
            st.markdown(f"""
            <div class="category-section">
                <h3>{category_name}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for gestures in this category
            cols = st.columns(min(len(gesture_indices), 4))
            
            for idx, gesture_idx in enumerate(gesture_indices):
                with cols[idx % 4]:
                    gesture_name = labels_dict[gesture_idx]
                    gesture_desc = gesture_descriptions.get(gesture_idx, "Gesture description")
                    
                    # Create gesture card
                    st.markdown(f"""
                    <div class="gesture-card">
                        <div class="gesture-index">{gesture_idx}</div>
                        <div class="gesture-title">{gesture_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Load and display gesture image
                    gesture_image = load_gesture_image(gesture_idx)
                    
                    if check_gesture_image(gesture_idx):
                        st.image(gesture_image, use_column_width=True, caption=f"Gesture: {gesture_name}")
                        st.success("✅ Image loaded")
                    else:
                        st.image(gesture_image, use_column_width=True, caption="Placeholder - Add image")
                        st.warning(f"⚠️ Add {gesture_idx}.jpg to images folder")
                    
                    # Gesture description
                    st.markdown(f"**Description:** {gesture_desc}")
                    
                    # Usage statistics
                    count = stats['gesture_counts'].get(gesture_name, 0)
                    if count > 0:
                        st.info(f"📊 Detected {count} times in this session")
                    
                    st.markdown("---")
        
        # Quick reference table
        st.markdown("### 📋 Quick Reference Table")
        
        # Create a comprehensive reference table
        reference_data = []
        for idx in range(20):
            gesture_name = labels_dict[idx]
            category = None
            for cat_name, indices in gesture_categories.items():
                if idx in indices:
                    category = cat_name
                    break
            
            image_status = "✅ Available" if check_gesture_image(idx) else "❌ Missing"
            usage_count = stats['gesture_counts'].get(gesture_name, 0)
            
            reference_data.append({
                "Index": idx,
                "Gesture": gesture_name,
                "Category": category,
                "Description": gesture_descriptions.get(idx, "")[:50] + "...",
                "Image Status": image_status,
                "Usage Count": usage_count
            })
        
        # Display as dataframe
        df = pd.DataFrame(reference_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download template
        st.markdown("### 📥 Download Setup Guide")
        
        if st.button("📋 Generate Image List Template"):
            template_text = "# Gesture Images Checklist\n\n"
            for idx in range(20):
                gesture_name = labels_dict[idx]
                template_text += f"□ {idx}.jpg - {gesture_name}\n"
            
            st.text_area("Copy this checklist:", template_text, height=300)
            st.info("Save this as a checklist and add corresponding images to your 'images' folder.")
    
    with tab4:
        st.markdown("### 📈 Advanced Analytics Dashboard")
        
        if st.session_state.session_stats['total_detections'] > 0:
            # Key metrics overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Detections", stats['total_detections'])
            with col2:
                st.metric("High Confidence Rate", f"{(stats['high_confidence_detections']/stats['total_detections']*100):.1f}%")
            with col3:
                st.metric("Average Confidence", f"{stats['avg_confidence']:.2f}")
            with col4:
                avg_processing_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence trend chart
                conf_chart = create_confidence_chart()
                if conf_chart:
                    st.plotly_chart(conf_chart, use_container_width=True)
            
            with col2:
                # Gesture frequency chart
                freq_chart = create_gesture_frequency_chart()
                if freq_chart:
                    st.plotly_chart(freq_chart, use_container_width=True)
            
            # Detailed statistics table
            if st.session_state.gesture_history:
                st.markdown("#### 📋 Detailed Detection Log")
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(st.session_state.gesture_history[-20:])  # Last 20 detections
                df['confidence_category'] = df['confidence'].apply(
                    lambda x: 'High (≥0.8)' if x >= 0.8 else 'Medium (0.6-0.8)' if x >= 0.6 else 'Low (<0.6)'
                )
                
                st.dataframe(
                    df[['timestamp', 'gesture', 'confidence', 'confidence_category']],
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("📊 Start detecting gestures to see analytics data here!")
    
    with tab5:
        st.markdown("### ℹ️ System Information")
        
        st.markdown("""
        #### 🚀 Advanced AI Sign Language Recognition System
        
        This professional-grade system provides comprehensive gesture recognition with detailed confidence analysis,
        real-time performance monitoring, and an interactive learning guide.
        
        ##### 🔧 Technical Specifications
        
        **AI/ML Components:**
        - **MediaPipe:** Google's advanced hand landmark detection
        - **Scikit-learn:** Machine learning classification engine
        - **OpenCV:** Computer vision processing pipeline
        - **NumPy:** High-performance numerical computing
        
        **Performance Metrics:**
        - **Processing Speed:** < 2 seconds per frame
        - **Accuracy:** Confidence-based quality assessment
        - **Gesture Support:** 20 distinct sign language gestures
        - **Multi-hand Support:** Up to 2 hands simultaneously
        
        **Enhanced Features:**
        - ✅ **Real-time confidence scoring** with visual indicators
        - ✅ **Interactive gesture guide** with visual learning aids
        - ✅ **Advanced analytics dashboard** with trend analysis
        - ✅ **Professional UI/UX** with responsive design
        - ✅ **Session management** with detailed statistics
        - ✅ **Batch processing** for uploaded images
        - ✅ **Multi-hand detection** with individual analysis
        - ✅ **Confidence-based color coding** for immediate feedback
        - ✅ **Performance monitoring** with processing time tracking
        - ✅ **Visual learning guide** with gesture demonstrations
        """)
        
        # System performance indicators
        st.markdown("##### 📊 Current Session Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Detection Quality**
            - High confidence threshold: ≥80%
            - Medium confidence threshold: 60-80%
            - Low confidence threshold: <60%
            """)
        
        with col2:
            st.markdown("""
            **⚡ Processing Performance**
            - Target processing time: <2s
            - Real-time analysis: ✅
            - Batch processing: ✅
            """)
        
        with col3:
            st.markdown("""
            **🔧 System Capabilities**
            - Multi-hand support: ✅
            - Confidence analysis: ✅
            - Analytics dashboard: ✅
            - Visual learning guide: ✅
            """)

if __name__ == "__main__":
    main()
