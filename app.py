import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import av

# Configure Streamlit Page Layout
st.set_page_config(page_title="Global Face Mesh App", layout="centered")
st.title("🤖 Real-Time Face Mesh (Legacy API)")
st.write("Using the classic mp.solutions API for compatibility with headless cloud environments.")

# Initialize legacy solution pieces
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceMeshProcessor(VideoProcessorBase):
    def __init__(self):
        # Setting up the legacy FaceMesh object
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to RGB for legacy processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the tessellation
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tessellation_style()
                )
                # Draw the main contours
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN server configurations
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="face-mesh-legacy",
    video_processor_factory=FaceMeshProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)