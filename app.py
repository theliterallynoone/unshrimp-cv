import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


#Ladies and gentlemen, the page setup

st.set_page_config(
    page_title="UnShrimp CV",
    page_icon="🪑",
    layout="centered"
)

st.title("🪑 UnShrimp CV")
st.caption("Sit up straight or face the consequences.")


# MediaPipe Pose Landmarker setup (ts so tuff)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"


landmarker_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6
)



# MediaPipe landmark indexes
# ----(because they said so)#
# MediaPipe Pose uses a fixed landmark ordering (unfortunalely)
#
# 7  = left ear
# 8  = right ear
# 11 = left shoulder
# 12 = right shoulder
#
# We use these four landmarks for the current posture measurement.
# ---(ill kms im so cooked)

LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# They call it the Session state

defaults = {
    "run": False,
    "calibrated": False,
    "baseline_angle": None,
    "calibration_samples": [],
    "calibration_start_time": None,
    "slouch_streak": 0,
    "last_alert_time": 0.0,
    "angle_history": deque(maxlen=150),
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



#Presenting upon thee, Controls

def start():
    st.session_state.run = True


def stop():
    st.session_state.run = False


def recalibrate():
    st.session_state.calibrated = False
    st.session_state.baseline_angle = None
    st.session_state.calibration_samples = []
    st.session_state.calibration_start_time = None
    st.session_state.slouch_streak = 0
    st.session_state.angle_history.clear()



# Sidebar
# ----(fun stuff)

with st.sidebar:
    st.header("Controls")

    col1, col2 = st.columns(2)

    col1.button(
        "▶ Start",
        on_click=start,
        use_container_width=True
    )

    col2.button(
        "⏹ Stop",
        on_click=stop,
        use_container_width=True
    )

    st.button(
        "🔄 Recalibrate",
        on_click=recalibrate,
        use_container_width=True
    )

    sensitivity = st.slider(
        "Sensitivity (degrees above baseline)",
        min_value=5,
        max_value=40,
        value=15,
        help="Lower = stricter. Flags slouching sooner."
    )

    consecutive_frames_needed = st.slider(
        "Frames before alert",
        min_value=1,
        max_value=15,
        value=5,
        help="Number of consecutive slouching frames required."
    )

    alert_cooldown = st.slider(
        "Cooldown between alerts (seconds)",
        min_value=1,
        max_value=15,
        value=5
    )

    camera_index = st.number_input(
        "Camera index",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )


# Geometry
#(hated this part, died doing it)

def ear_shoulder_angle_from_vertical(shoulder_xy, ear_xy):
    """
    Calculate the angle of the shoulder → ear vector
    relative to the vertical direction.

    0 degrees means the ear is directly above the shoulder.
    """

    dx = ear_xy[0] - shoulder_xy[0]

    # Image y coordinates increase downward, so we flip the y direction.
    dy = shoulder_xy[1] - ear_xy[1]

    angle_rad = math.atan2(dx, dy)

    return math.degrees(angle_rad)

# Selecting the more visible side because well im lazy and this is easier than checking which side is facing the camera

def pick_visible_side(landmarks, width, height):

    left_visibility = (
        landmarks[LEFT_SHOULDER].visibility
        + landmarks[LEFT_EAR].visibility
    )

    right_visibility = (
        landmarks[RIGHT_SHOULDER].visibility
        + landmarks[RIGHT_EAR].visibility
    )

    if left_visibility >= right_visibility:

        shoulder = landmarks[LEFT_SHOULDER]
        ear = landmarks[LEFT_EAR]

    else:

        shoulder = landmarks[RIGHT_SHOULDER]
        ear = landmarks[RIGHT_EAR]

    shoulder_xy = (
        shoulder.x * width,
        shoulder.y * height
    )

    ear_xy = (
        ear.x * width,
        ear.y * height
    )

    return shoulder_xy, ear_xy


# Alert!!!
# --(its d-time)--

def trigger_slouch_alert():

    st.session_state.last_alert_time = time.time()

    st.toast(
        "😱 SLOUCH DETECTED",
        icon="⚠️"
    )


 
# Streamlit placeholders
 

video_placeholder = st.empty()
status_placeholder = st.empty()
chart_placeholder = st.empty()


 
# Main camera loopie
 

if st.session_state.run:

    cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():

        st.error(
            f"Couldn't open camera index {camera_index}. "
            "Try a different index or check camera permissions."
        )

        st.session_state.run = False

    else:

        with PoseLandmarker.create_from_options(
            landmarker_options
        ) as landmarker:

            while st.session_state.run:

                ok, frame = cap.read()

                if not ok:

                    st.warning("Lost camera feed.")

                    break

                # Mirror the webcam
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                # OpenCV uses BGR.
                # MediaPipe expects an RGB image here.
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert the NumPy image into a MediaPipe Image.
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # Run pose detection.
                results = landmarker.detect(mp_image)

                angle = None

                # ------------------------------------------------------
                # If a person was detected
                # ------------------------------------------------------

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks[0]

                    shoulder_xy, ear_xy = pick_visible_side(
                        landmarks,
                        width,
                        height
                    )

                    angle = ear_shoulder_angle_from_vertical(
                        shoulder_xy,
                        ear_xy
                    )

                    # Draw the selected shoulder → ear line.
                    cv2.line(
                        frame,
                        (
                            int(shoulder_xy[0]),
                            int(shoulder_xy[1])
                        ),
                        (
                            int(ear_xy[0]),
                            int(ear_xy[1])
                        ),
                        (0, 255, 255),
                        3
                    )

                    # Draw the two important points.
                    cv2.circle(
                        frame,
                        (
                            int(shoulder_xy[0]),
                            int(shoulder_xy[1])
                        ),
                        8,
                        (0, 255, 255),
                        -1
                    )

                    cv2.circle(
                        frame,
                        (
                            int(ear_xy[0]),
                            int(ear_xy[1])
                        ),
                        8,
                        (0, 255, 255),
                        -1
                    )

                # ------------------------------------------------------
                # Calibration
                # ------------------------------------------------------

                if angle is not None and not st.session_state.calibrated:

                    if st.session_state.calibration_start_time is None:

                        st.session_state.calibration_start_time = time.time()

                    elapsed = (
                        time.time()
                        - st.session_state.calibration_start_time
                    )

                    st.session_state.calibration_samples.append(angle)

                    if elapsed < 3.0:

                        status_placeholder.info(
                            f"📏 Calibrating... sit up straight! "
                            f"({3.0 - elapsed:.1f}s left)"
                        )

                    else:

                        st.session_state.baseline_angle = float(
                            np.median(
                                st.session_state.calibration_samples
                            )
                        )

                        st.session_state.calibrated = True

                # ------------------------------------------------------
                # Posture detection
                # ------------------------------------------------------

                elif angle is not None and st.session_state.calibrated:

                    st.session_state.angle_history.append(angle)

                    threshold = (
                        st.session_state.baseline_angle
                        + sensitivity
                    )

                    is_slouching_now = angle > threshold

                    if is_slouching_now:

                        st.session_state.slouch_streak += 1

                    else:

                        st.session_state.slouch_streak = 0

                    confirmed_slouch = (
                        st.session_state.slouch_streak
                        >= consecutive_frames_needed
                    )

                    cooldown_ok = (
                        time.time()
                        - st.session_state.last_alert_time
                        > alert_cooldown
                    )

                    if confirmed_slouch and cooldown_ok:

                        trigger_slouch_alert()

                    label = (
                        "SLOUCHING"
                        if confirmed_slouch
                        else "Good posture"
                    )

                    status_placeholder.markdown(
                        f"**Status:** "
                        f"{'🔴 Slouching' if confirmed_slouch else '🟢 Good posture'} "
                        f"&nbsp;&nbsp; "
                        f"angle = `{angle:.1f}°` "
                        f"&nbsp; "
                        f"threshold = `{threshold:.1f}°`"
                    )

                    if len(st.session_state.angle_history) > 1:

                        chart_placeholder.line_chart(
                            list(st.session_state.angle_history)
                        )

                else:

                    status_placeholder.warning(
                        "No person detected — "
                        "make sure you're in frame."
                    )

                # Show the camera frame.
                video_placeholder.image(
                    frame,
                    channels="BGR",
                    use_container_width=True
                )

        cap.release()

else:

    st.info(
        "Press **▶ Start** in the sidebar to begin. "
        "You'll calibrate for 3 seconds first — sit up straight!"
    )