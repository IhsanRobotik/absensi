from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize the webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(1)

# Set resolution to mimic ESP32-CAM (e.g., 320x240 for low-res streaming)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the frame in the MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask server on localhost, port 81 to mimic ESP32-CAM default port
    app.run(host='0.0.0.0', port=81, threaded=True)

# Release the webcam when the program exits
cap.release()