import os
import cv2
import torch
import time
import math
import webbrowser
import streamlit as st
import numpy as np
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from emotion_net import EmotionNet
from movie_utils import get_most_shown_emotion, open_url_based_on_emotion

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
device = "cuda" if torch.cuda.is_available() else "cpu"
current_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_path)
model_path = os.path.join(current_path, 'models/emotion_recognition_model.pth')

model = EmotionNet().to(device)
model.load_state_dict(torch.load(model_path, map_location = device, weights_only = True))
model.eval()

st.title("Movie Recommendation System based on Emotions")
st.write("This app detects emotions in real-time from your webcam.")

list_of_lang = ["English", "Telugu", "Malayalam", "Tamil", "Hindi"]
selected_lang = st.selectbox("Choose your preferred language for viewing the movie:", list_of_lang)

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()
        self.flag = True
        self.emotion_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        self.labels_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        img = frame.to_ndarray(format="bgr24")
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sub_face_img = gray[y:y + h, x:x + w]
        
            resized = cv2.resize(sub_face_img, (48, 48))
        
            normalized = np.divide(resized, 255.0)
            reshaped = normalized.reshape(1, 1, 48, 48) 

            input_tensor = torch.tensor(reshaped, dtype=torch.float32).to(device)
        
            with torch.no_grad():
                result = model(input_tensor)
                emotion = int(torch.argmax(result, dim=1).item())

            self.emotion_counter[emotion] += 1
            emotion_str = self.labels_dict[emotion]

            if self.flag:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(img, emotion_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if math.floor(elapsed_time) >= 20:
                    self.flag = False
                    result_emotion = get_most_shown_emotion(self.emotion_counter, self.labels_dict)
                    url = open_url_based_on_emotion(result_emotion, selected_lang)
                    webbrowser.open(url)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": "turn:turn.anyfirewall.com:443?transport=tcp", "credential": "webrtc", "username": "webrtc"},
        ]
    }
)
