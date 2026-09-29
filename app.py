import streamlit as st
import cv2
import mediapipe as mp

st.title("unshrimp-cv")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

camera = cv2.VideoCapture(0)

ret, frame = camera.read()

if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)
        st.write(results)

    st.image(frame_rgb)

camera.release()