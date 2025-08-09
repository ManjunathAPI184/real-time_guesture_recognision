import streamlit as st
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model()
        self.mp_hands, self.mp_drawing, self.mp_drawing_styles = init_mediapipe()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.5
        )
        self.current_prediction = "No gesture detected"

    def update_confidence(self, confidence):
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=confidence,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        self.current_prediction = "No gesture detected"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                H, W, _ = img.shape

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

                if self.model and len(data_aux) == 42:
                    try:
                        prediction = self.model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
                        self.current_prediction = predicted_character

                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                        cv2.putText(img, predicted_character, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        self.current_prediction = f"Error: {str(e)[:20]}..."

                self.mp_drawing.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("🤟 Real-Time Sign Language Recognition")

    detection_confidence = st.sidebar.slider(
        "Detection Confidence", 0.1, 1.0, 0.3, 0.1
    )

    ctx = webrtc_streamer(
        key="sign-language-detection",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"style": {"width": "100%", "margin": "0 auto"}}
    )

    if ctx.video_transformer:
        ctx.video_transformer.update_confidence(detection_confidence)
        st.sidebar.write("Prediction:", ctx.video_transformer.current_prediction)

if __name__ == "__main__":
    main()
