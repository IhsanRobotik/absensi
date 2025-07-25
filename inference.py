import face_recognition
import cv2
import pickle
import os

FACES_DIR = "faces"
ENCODINGS_FILE = "encodings.pkl"
MODEL = "hog"

print(f"[*] Loading encodings from '{ENCODINGS_FILE}'...")
if not os.path.exists(ENCODINGS_FILE):
    print(f"[!] Error: Encodings file not found. Please run train.py first.")
    exit()

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("[*] Starting video stream...")
url = "http://192.168.1.26:81/stream"
video_capture = cv2.VideoCapture(1)
if not video_capture.isOpened():
    print("[!] Error: Could not open webcam.")
    exit()

video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[*] Press 'q' to quit.")

frame_skip = 2
frame_count = 0

while True:
    for _ in range(frame_skip):
        video_capture.grab()
    frame_count += frame_skip

    ret, frame = video_capture.retrieve()
    if not ret:
        print("[!] Error: Failed to grab frame.")
        break

    if frame_count % frame_skip == 0:
        face_locations = face_recognition.face_locations(frame, model=MODEL)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            print(name)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[*] Video stream stopped.")