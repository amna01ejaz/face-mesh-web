import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
import urllib.request
import os
import tempfile

st.set_page_config(page_title="Global Face Mesh App", layout="centered")
st.title("🤖 Real-Time Face Mesh")
st.write("This app uses the modern MediaPipe Tasks API to map face landmarks via your browser.")

# Writable directory for downloading the model
TEMP_DIR = tempfile.gettempdir()
MODEL_PATH = os.path.join(TEMP_DIR, "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading face mesh model file... Please wait."):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.face_landmarks:
            h, w, _ = img.shape
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="face-mesh",
    video_processor_factory=FaceMeshProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)