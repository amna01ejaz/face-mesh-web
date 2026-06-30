import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
import urllib.request
import os

# Configure Streamlit Page Layout
st.set_page_config(page_title="Global Face Mesh App", layout="centered")
st.title("🤖 Real-Time Face Mesh (Modern API)")
st.write("This app uses the modern MediaPipe Tasks API to map face landmarks via your browser.")

# Step 1: Download the Face Landmarker model asset if it doesn't exist
import tempfile
MODEL_PATH = os.path.join(tempfile.gettempdir(), "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading face mesh model file... Please wait."):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

# Define the modern video frame processor class
class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        # Configure the modern Landmarker options
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
        
        # Convert frame to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        # Run face landmark detection
        detection_result = self.detector.detect(mp_image)
        
        # Draw landmarks if a face is found
        if detection_result.face_landmarks:
            h, w, _ = img.shape
            for face_landmarks in detection_result.face_landmarks:
                for landmark in face_landmarks:
                    # Convert normalized coordinates (0.0 to 1.0) back to pixel coordinates
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Draw subtle green dots for the mesh
                    cv2.circle(img, (cx, cy), 1, (0, 255, 0), -1)
                    
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Set up free public STUN servers for network connections
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Render the web stream interface
webrtc_streamer(
    key="face-mesh",
    video_processor_factory=FaceMeshProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
)