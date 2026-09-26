import streamlit as st
import cv2
st.title("Unshrimp-CV")
st.write("Posture detection, but make it less shrimp.")
cam = cv2.VideoCapture(0)
ret, frame = cam.read()
if ret:
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame)
cam.release()