# =========================
# FINAL VERSION — SINGLE BUZZER + 5-MIN COOLDOWN + TERMINAL LOGS
# =========================

import logging
logging.getLogger().setLevel(logging.ERROR)

import os
os.environ["ULTRALYTICS_ONNX"] = "0"
os.environ["YOLO_ONNX"] = "0"
print("CHECKPOINT: ONNX Disabled!")

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque

# AUDIO COMPONENTS
import threading
from playsound import playsound
from pydub import AudioSegment
import ffmpeg

# EVENT STORAGE
ALERT_EVENTS = []

# SINGLE BUZZER SOUND
BUZZER_AUDIO = "announcement.wav"

# CONFIG
INPUT_VIDEO = "CCTVFootage.mp4"
OUTPUT_VIDEO = "output.avi"
LINE_MIN_LENGTH = 80

print("\n==== LOADING MODELS ====")
yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)
print("MODELS LOADED SUCCESSFULLY.\n")


# ============================================================
# PERSON DETECTION
# ============================================================
def detect_people(frame):
    results = yolo(frame, classes=[0], verbose=False)
    persons = []

    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        persons.append((xyxy.tolist(), conf))

    print(f" - People detected: {len(persons)}")
    return persons


# ============================================================
# TRAIN DETECTION
# ============================================================
def detect_train(frame):
    results = yolo(frame, classes=[6], verbose=False)

    if results and results[0].boxes:
        box = results[0].boxes[0]
        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
        conf = float(box.conf[0])
        print(" - TRAIN detected")
        return (bbox, conf)

    print(" - TRAIN not detected")
    return None


# ============================================================
# LINE DETECTION (Yellow Line)
# ============================================================
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 60, 60])
    upper_yellow = np.array([45, 255, 255])
    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.medianBlur(mask, 5)

    lines = cv2.HoughLinesP(mask, 1, np.pi/180, 60,
                            minLineLength=LINE_MIN_LENGTH, maxLineGap=20)

    if lines is not None:
        print(" - LINE detected")
        longest = max(lines, key=lambda L: np.hypot(L[0][2]-L[0][0], L[0][3]-L[0][1]))
        return longest[0]

    print(" - LINE not detected")
    return None


# ============================================================
# DISTANCE TO LINE
# ============================================================
def point_to_segment_distance(px, py, x1, y1, x2, y2):
    A = np.array([px, py])
    B = np.array([x1, y1])
    C = np.array([x2, y2])
    AB = C - B

    t = np.dot(A-B, AB) / (np.dot(AB, AB) + 1e-6)
    t = max(0, min(1, t))
    projection = B + t * AB

    return np.linalg.norm(A - projection)


# ============================================================
# RISK ASSESSMENT
# ============================================================
def compute_risks(persons, train, line, frame_time):
    risks = {}

    for pid, tlbr, conf in persons:
        x1, y1, x2, y2 = tlbr
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if line is not None:
            dist = point_to_segment_distance(cx, cy, *line)
        else:
            dist = 9999

        if dist < 30:
            status = "CRITICAL"
        elif dist < 60:
            status = "WARNING"
        else:
            status = "SAFE"

        risks[pid] = {"distance": dist, "status": status}

    return risks


# ============================================================
# OVERLAY DRAWING
# ============================================================
def overlay(frame, persons, train, line, risks):

    if line is not None:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)

    if train is not None:
        x1, y1, x2, y2 = train[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "TRAIN", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    for pid, tlbr, conf in persons:
        x1, y1, x2, y2 = map(int, tlbr)
        status = risks[pid]["status"]

        if status == "SAFE":
            color = (0, 255, 0)
        elif status == "WARNING":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, status, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# ============================================================
# MAIN PROGRAM
# ============================================================
def main():

    print("\n==== OPENING VIDEO ====")
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS={fps}, Resolution={W}x{H}")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps, (W, H)
    )

    frame_num = 0

    print("\n==== PROCESSING STARTED ====\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n==== VIDEO FINISHED ====")
            break

        print(f"\n---- FRAME {frame_num} ----")

        persons_raw = detect_people(frame)
        train = detect_train(frame)
        line = detect_line(frame)

        tracker_inputs = []
        for (x1, y1, x2, y2), conf in persons_raw:
            w = x2 - x1
            h = y2 - y1
            tracker_inputs.append(([x1, y1, w, h], conf))

        tracks = tracker.update_tracks(tracker_inputs, frame=frame)
        persons = [(t.track_id, t.to_tlbr(), t.det_conf)
                   for t in tracks if t.is_confirmed()]

        risks = compute_risks(persons, train, line, frame_num / fps)

        current_time = frame_num / fps

        # LOG RISK EVENTS
        for pid in risks:
            status = risks[pid]["status"]

            if status == "CRITICAL":
                ALERT_EVENTS.append(("CRITICAL", current_time))
                print(" !!! CRITICAL EVENT DETECTED")
                break

            elif status == "WARNING":
                ALERT_EVENTS.append(("WARNING", current_time))
                print(" ! WARNING EVENT DETECTED")
                break

        overlay(frame, persons, train, line, risks)

        out.write(frame)
        frame_num += 1

    print("\n==== GENERATING FINAL AUDIO TRACK ====\n")

    # ============================================================
    # AUDIO GENERATION WITH 5-MINUTE COOLDOWN
    # ============================================================

    video_duration = frame_num / fps
    silent_audio = AudioSegment.silent(duration=int(video_duration * 1000))

    buzzer = AudioSegment.from_wav(BUZZER_AUDIO)
    buzzer_len = len(buzzer)

    final_audio = silent_audio

    print(f"Total Alert Events Captured: {len(ALERT_EVENTS)}")

    # 5 minutes = 300 seconds
    cooldown_ms = 5 * 1000

    cleaned_events = []
    last_buzzer_time_ms = -999999999

    print("Applying 5-minute cooldown filtering...")

    for alert_type, ts in ALERT_EVENTS:

        event_time_ms = int(ts * 1000)

        # PRIORITY: CRITICAL overrides WARNING at same timestamp
        if cleaned_events and cleaned_events[-1][1] == ts:
            if alert_type == "CRITICAL":
                cleaned_events[-1] = ("CRITICAL", ts)
            continue

        # 5-minute cooldown enforcement
        if event_time_ms < last_buzzer_time_ms + cooldown_ms:
            print(f" - Skipped (cooldown active): {alert_type} at {ts:.2f}s")
            continue

        cleaned_events.append((alert_type, ts))
        last_buzzer_time_ms = event_time_ms
        print(f" + Accepted buzzer at {ts:.2f}s")

    print(f"\nFinal Alert Count After Cooldown = {len(cleaned_events)}\n")

    # Overlay buzzer
    for alert_type, ts in cleaned_events:
        print(f"Overlaying buzzer at {ts:.2f}s...")
        final_audio = final_audio.overlay(buzzer - 3, position=int(ts * 1000))

    final_audio = final_audio.set_channels(2)
    final_audio.export("alerts_audio.wav", format="wav")

    print("\n==== MERGING VIDEO + AUDIO ====\n")

    video_stream = ffmpeg.input(OUTPUT_VIDEO)
    audio_stream = ffmpeg.input("alerts_audio.wav")

    ffmpeg.output(
        video_stream,
        audio_stream,
        "final_output.mp4",
        vcodec="copy",
        acodec="aac",
        audio_bitrate="192k",
        ar=44100,
        ac=2
    ).overwrite_output().run()

    print("\n==============================")
    print(" FINAL VIDEO READY: final_output.mp4")
    print("==============================\n")


if __name__ == "__main__":
    main()
