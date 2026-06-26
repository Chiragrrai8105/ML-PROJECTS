import streamlit as st
import pandas as pd
import os
import cv2
import av
import plotly.express as px
import pandas as pd
import time
import threading
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from detector import SeniorCitizenDetector
from utils import CSVLogger
import torch
import platform
import psutil
from datetime import datetime

st.sidebar.markdown("---")

st.sidebar.write("Current Time")

st.sidebar.success(
    datetime.now().strftime("%d-%m-%Y %H:%M:%S")
)

@st.cache_resource
def load_detector():
    return SeniorCitizenDetector()

detector = load_detector()

logger = CSVLogger("visits.csv")

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.detector = detector

        self.logger = logger

        self.total = 0
        self.senior = 0
        self.male = 0
        self.female = 0

        self.fps = 0
        self.prev_time = time.time()
        self.latest_frame = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img, detections = self.detector.detect(img)
        self.latest_frame = img.copy()

        self.total = len(detections)
        self.senior = 0
        self.male = 0
        self.female = 0

        for det in detections:

            if det["gender"] == "Male":
                self.male += 1
            else:
                self.female += 1

            if det["status"] == "Senior Citizen":

                self.senior += 1

                self.logger.log(
                    det["age"],
                    det["gender"],
                    det["status"]
                )

        current = time.time()

        self.fps = int(1 / (current - self.prev_time))

        self.prev_time = current

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )
# ---------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Senior Citizen Identification System",
    page_icon="👴",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color:#f5f7fa;
}

section[data-testid="stSidebar"]{
    background:#1e293b;
}

section[data-testid="stSidebar"] *{
    color:white;
}

div[data-testid="metric-container"]{
    border-radius:15px;
    padding:18px;
    background:#ffffff;
    box-shadow:0 2px 10px rgba(0,0,0,.12);
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

h1,h2,h3{
    color:#1e293b;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📷 Live Detection",
        "📊 Detection History",
        "📈 Analytics",
        "⚙ Settings",
        "ℹ About"
    ]
)

# ---------------------------------------------------
# HOME
# ---------------------------------------------------

if page == "🏠 Home":

    st.title("👴 Senior Citizen Identification System")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
This AI-powered system detects people from a live camera feed,
predicts their age and gender, identifies senior citizens
(age ≥ 60), and stores the visit details automatically.

### Features

- ✅ Face Detection (YOLO)
- ✅ Age Prediction
- ✅ Gender Prediction
- ✅ Senior Citizen Detection
- ✅ Automatic CSV Logging
- ✅ Analytics Dashboard
    """)

    st.info("Click **Live Detection** from the sidebar to start the camera.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "YOLO + EfficientNet")

    with col2:
        st.metric("Age Limit", "60+")

    with col3:
        st.metric("Status", "Ready")

# ---------------------------------------------------
# LIVE DETECTION
# ---------------------------------------------------

elif page == "📷 Live Detection":

    st.title("📷 Live Detection")

    st.markdown("---")

    col1, col2 = st.columns([3,1])

    with col1:

        ctx = webrtc_streamer(

            key="senior-camera",

            video_processor_factory=VideoProcessor,

            media_stream_constraints={

                "video": True,

                "audio": False

            },

            async_processing=True
        )

    with col2:

        st.subheader("Live Statistics")

        if ctx.video_processor:

            vp = ctx.video_processor

            st.metric("Persons", vp.total)

            st.metric("Senior", vp.senior)

            st.metric("Male", vp.male)

            st.metric("Female", vp.female)

            st.metric("FPS", vp.fps)

        else:

            st.info("Start the camera.")

    if st.button("📸 Capture Frame"):

        if ctx.video_processor and ctx.video_processor.latest_frame is not None:

            os.makedirs("screenshots", exist_ok=True)

            cv2.imwrite(

                f"screenshots/frame_{time.time()}.jpg",

                ctx.video_processor.latest_frame

            )

            st.success("Screenshot Saved")

        else:

            st.warning("Start the camera before capturing a frame.")            

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------

elif page == "📊 Detection History":

    st.title("📊 Detection History")

    csv_file = "visits.csv"

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)

        if df.empty:
            st.info("No detections available yet.")
        else:

            st.subheader("Search Records")

            search = st.text_input("Search")

            gender = st.selectbox(
                "Gender",
                ["All", "Male", "Female"]
            )

            status = st.selectbox(
                "Status",
                ["All", "Senior Citizen"]
            )

            filtered = df.copy()

            if search:
                filtered = filtered[
                    filtered.astype(str)
                    .apply(lambda x: x.str.contains(search, case=False))
                    .any(axis=1)
                ]

            if gender != "All":
                filtered = filtered[
                    filtered["Gender"] == gender
                ]

            if status != "All":
                filtered = filtered[
                    filtered["Status"] == status
                ]

            st.dataframe(
                filtered,
                use_container_width=True
            )

            st.download_button(
                "⬇ Download CSV",
                filtered.to_csv(index=False),
                "visits.csv",
                "text/csv"
            )

    else:

        st.warning("No CSV file found.")

# ---------------------------------------------------
# ANALYTICS
# ---------------------------------------------------

elif page == "📈 Analytics":

    st.title("📈 Analytics Dashboard")

    csv_file = "visits.csv"

    if os.path.exists(csv_file):

        df = pd.read_csv(csv_file)

        if df.empty:

            st.info("No data available.")

        else:

            total = len(df)

            male = len(df[df["Gender"]=="Male"])

            female = len(df[df["Gender"]=="Female"])

            senior = len(df)

            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Total Visits", total)

            c2.metric("Senior Citizens", senior)

            c3.metric("Male", male)

            c4.metric("Female", female)

            st.divider()

            st.subheader("Gender Distribution")

            fig = px.pie(
                df,
                names="Gender",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Age Distribution")

            fig2 = px.histogram(
                df,
                x="Age",
                nbins=20
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.subheader("Status Distribution")

            fig3 = px.bar(
                df["Status"]
                .value_counts()
                .reset_index(),
                x="Status",
                y="count"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            st.subheader("Recent Detections")

            st.dataframe(
                df.tail(10),
                use_container_width=True
            )

    else:

        st.warning("No CSV available.")

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

elif page == "⚙ Settings":

    st.title("⚙ Settings")

    st.markdown("---")

    confidence = st.slider(
        "Confidence Threshold",
        0.10,
        1.00,
        0.45
    )

    resolution = st.selectbox(
        "Camera Resolution",
        [
            "640x480",
            "1280x720"
        ]
    )

    save_images = st.checkbox(
        "Save Detected Faces",
        value=False
    )

# ---------------------------------------------------
# ABOUT
# ---------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ About")

    st.markdown("---")

    st.write("""
## Senior Citizen Identification System

Version **1.0**

### Technologies Used

- Streamlit
- YOLO Face Detection
- EfficientNet-B0
- PyTorch
- OpenCV
- Pandas

Developed for AI-based real-time senior citizen identification.
""")
    