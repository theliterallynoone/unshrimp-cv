# UnShrimp CV - Field Notes

## 1. What are we (me and claude lmao) building?

UnShrimp CV is a computer-vision posture detector built using:

- Python
- Streamlit
- OpenCV
- MediaPipe Pose Landmarker

The goal is to detect when the user's sitting posture changes from their normal upright posture.

---

## 2. How the detector works

The basic pipeline is:

    Webcam
       ↓
    OpenCV
       ↓
    MediaPipe Pose Landmarker
       ↓
    Body landmarks
       ↓
    Geometry calculations
       ↓
    Posture classification
       ↓
    Streamlit UI

MediaPipe gives us body landmarks with normalized coordinates.

For posture detection, we are mainly interested in landmarks around:

- ears
- shoulders

---

## 3. Posture geometry

The current idea is to measure the angle of the vector from the shoulder to the ear.

The angle is measured relative to the vertical direction.

Conceptually:

          ear
           ●
          /
         /
        ● shoulder

If the ear is almost directly above the shoulder, the angle is small.

If the head moves forward/down relative to the shoulder, the angle increases.

The detector therefore uses the ear-to-shoulder angle as one measurement of posture.

---

## 4. Why calibration?

We should not assume that one angle works for every person.

Different people have:

- different body proportions
- different natural sitting positions
- different camera positions
- different camera angles

Instead, the program can calibrate when the user sits upright.

During calibration:

1. User sits in their normal upright position.
2. The program records the angle for a few seconds.
3. A baseline angle is calculated.
4. The live angle is compared against that baseline.

Conceptually:

    slouch threshold = baseline angle + sensitivity

If the live angle exceeds the threshold for several consecutive frames, the program considers the user to be slouching.

---

## 5. False-positive protection

A single strange frame should not immediately trigger an alert.

Instead, the detector can require several consecutive frames to indicate slouching.

There can also be a cooldown between alerts.

This gives us:

    detected slouch
          ↓
    consecutive frames?
          ↓
       confirmed
          ↓
    cooldown finished?
          ↓
        alert

---

## 6. Planned features

Current:

- webcam input
- pose detection
- posture angle
- calibration
- slouch detection

Later:

- posture visualization
- live posture graph
- posture score
- slouch statistics
- sound alert
- better posture features
- testing under different camera angles

---

## 7. Important implementation note

This project uses the newer MediaPipe Tasks API rather than the older `mp.solutions.pose` API.

The pose model is stored locally as:

    pose_landmarker_lite.task

---

## 8. Things to investigate

- How reliable is ear-to-shoulder angle by itself?
- What happens when the user turns sideways?
- What happens when only one ear is visible?
- How much does camera placement affect the measurement?
- How should calibration handle movement?
- Would additional landmarks improve the classifier?