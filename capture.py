import cv2
import os
import time

# Directory to save the face images
FACES_DIR = "faces"
# Number of images to capture for each person
NUM_IMAGES = 20

def capture_faces():
    count = 0
    person_name = input("Enter the name of the person: ")
    if not person_name:
        print("[!] Error: Name cannot be empty.")
        return

    person_dir = os.path.join(FACES_DIR, person_name)

    # Create a directory for the person if it doesn't exist
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"[*] Directory created for '{person_name}' at '{person_dir}'")
    else:
        print(f"[*] Directory for '{person_name}' already exists.")

    # Initialize the webcam
    video_capture = cv2.VideoCapture(1)
    if not video_capture.isOpened():
        print("[!] Error: Could not open webcam.")
        return

    while True:
        ret, frame = video_capture.read()
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xff == ord('c'):
            # Save the captured image
            image_path = os.path.join(person_dir, f"{count + 1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"    - Captured image {count + 1}/{NUM_IMAGES}")
            count += 1
            time.sleep(0.5)  # Wait 0.5 seconds between captures


        # Hit 'q' on the keyboard to quit early!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print(f"[*] Face capture complete. {count} images saved.")

if __name__ == "__main__":
    capture_faces()
