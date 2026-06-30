import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av

# Initialize MediaPipe Face Mesh solutions
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configure Streamlit Page Layout
st.set_page_config(page_title="Global Face Mesh App", layout="centered")
st.title("🤖 Real-Time Face Mesh")
st.write("This app uses MediaPipe to map face landmarks. It runs entirely via your browser.")

# Define the video frame processor class
class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize face mesh with optimized parameters for real-time tracking
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert incoming WebRTC frame to a NumPy array (BGR format for OpenCV)
        img = frame.to_ndarray(format="bgr24")
        
        # Flip the image horizontally for a natural mirror effect
        img = cv2.flip(img, 1)
        
        # Convert to RGB as required by MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        # If landmarks are detected, draw the mesh networks onto the image
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Draw the delicate tessellation lines
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tessellation_style()
                )
                # 2. Draw the main contours (eyes, brows, lips, face oval)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                # 3. Draw iris landmarks
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )
                
        # Return the processed frame back to the browser stream
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Set up free public STUN servers so devices behind firewalls can connect smoothly
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# Render the video stream UI element
webrtc_streamer(
    key="face-mesh",
    video_processor_factory=FaceMeshProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)