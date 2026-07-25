import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from detector import detect

# Page settings
st.set_page_config(
    page_title="Right & Left Hand Detection",
    page_icon="✋",
    layout="wide"
)

# Title
st.title("✋ Right & Left Hand Detection")
st.markdown("### Real-Time Hand Detection using OpenCV + MediaPipe")

st.write(
    """
This application detects whether your hand is **Left** or **Right**
using your webcam in real time.
"""
)

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.markdown("""
### Technologies Used

- Python
- OpenCV
- MediaPipe
- Streamlit

### Instructions

1. Click **START**
2. Allow camera permission
3. Show your hand
4. Press **STOP** when finished
""")

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = detect(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Webcam with STUN server configuration
webrtc_streamer(
    key="hand-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)