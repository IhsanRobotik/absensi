
import face_recognition
import pickle
import os

# Path to the directory containing face images
FACES_DIR = "faces"
# File to save the face encodings
ENCODINGS_FILE = "encodings.pkl"

def train_model():
    """
    Trains the face recognition model by creating face encodings from images.
    """
    known_encodings = []
    known_names = []

    print(f"[*] Starting training using images in '{FACES_DIR}'...")

    # Loop through each person in the faces directory
    for person_name in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person_name)

        # Skip if it's not a directory
        if not os.path.isdir(person_dir):
            continue

        print(f"    - Training on '{person_name}'")
        # Loop through each image of the person
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            # Load the image
            try:
                image = face_recognition.load_image_file(image_path)
                # Find face encodings
                face_encodings = face_recognition.face_encodings(image)

                # Ensure at least one face was found
                if face_encodings:
                    # Use the first encoding found
                    encoding = face_encodings[0]
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                else:
                    print(f"      [!] Warning: No face found in {image_name}. Skipping.")
            except Exception as e:
                print(f"      [!] Error processing {image_name}: {e}")

    # Save the encodings to a file
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print(f"[*] Training complete. Encodings saved to '{ENCODINGS_FILE}'")

if __name__ == "__main__":
    train_model()