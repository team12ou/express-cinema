import time
import math
import random
import sys
import cv2
import webbrowser
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from keras.models import load_model
import av
import os

# Error logging
sys.stderr = open('error.log', 'w')

# Loading model
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
current_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_path)
current_path = os.path.join(current_path, 'model_file.h5')
model = load_model('./model_file.h5')

# Creating website
st.title("Movie Recommendation System based on emotions")
st.write("This app detects emotions in real-time from your webcam.")
list_of_lang = ["English", "Telugu", "Malayalam", "Tamil", "Hindi"]
selected_lang = st.selectbox("Choose your preferred language for viewing the movie:", list_of_lang)

# Image Processing
class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()
        self.count = 0
        self.emotion_counter = {0: 0, 1: 0, 2: 0, 3: 0}
        self.labels_dict = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad"}
    
    # Function to return emotion that was shown most
    def get_emotion(self):
        max_key = max(self.emotion_counter, key=lambda x: self.emotion_counter[x])
        return self.labels_dict[max_key]

    # Function to return genre parameter
    def get_genre(self, emotion):
        genre_dict = {
            "Angry": ["comedy", "family", "romance"],
            "Happy": ["horror", "sci-fi", "mystery"],
            "Neutral": ["action", "adventure", "drama"],
            "Sad": ["sport", "thriller", "crime"]
        }

        if str(selected_lang) == "English":
            genre_dict["Neutral"].append("western")
            genre_dict["Sad"].append("animation")

        result_list = genre_dict[emotion]
        result_genre = random.choice(result_list)
        result = "&genres=" + result_genre
        return result

    # Function to return language parameter
    def get_lang(self):
        lang_dict = {"English": "en", "Telugu": "te", "Malayalam": "ml", "Tamil": "ta", "Hindi": "hi"}
        result = "&primary_language=" + lang_dict[str(selected_lang)]
        return result

    # Function to redirect to IMDb
    def open_url_based_on_emotion(self, emotion):
        base_url = "https://www.imdb.com/search/title/?title_type=feature&user_rating=7,10"
        genre = self.get_genre(emotion)
        lang = self.get_lang()
        sort_results = "&sort=num_votes,desc"
        result_url = base_url + genre + lang + sort_results
        return result_url

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            self.emotion_counter[label] += 1
            emotion = self.labels_dict[label]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
            if math.floor(elapsed_time) >= 25 and self.count < 1:
                self.count += 1
                result_emotion = self.get_emotion()
                url = self.open_url_based_on_emotion(result_emotion)
                webbrowser.open(url)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        
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
