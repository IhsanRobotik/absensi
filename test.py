import face_recognition
import cv2

# Load the debug image
image = face_recognition.load_image_file("debug_frame.jpg")

# Find face locations
face_locations = face_recognition.face_locations(image)

if face_locations:
    print(f"[*] Found {len(face_locations)} face(s) in the debug image.")
    # Optionally, draw rectangles on the image to visualize
    # for (top, right, bottom, left) in face_locations:
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    # cv2.imwrite("debug_frame_with_faces.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # print("[*] Saved debug_frame_with_faces.jpg")
else:
    print("[*] No faces found in the debug image.")
