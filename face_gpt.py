import cv2
import numpy as np
import os
import time

# Load the pre-trained models for face detection and recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to prepare training data and labels
def prepare_training_data(folder_path):
    faces = []
    labels = []
    names_dict = {}
    label = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)
            name = os.path.splitext(filename)[0]
            names_dict[label] = name
            label += 1
    return faces, labels, names_dict

# Training the recognizer
faces, labels, names_dict = prepare_training_data('path_to_faces_folder')
recognizer.train(faces, np.array(labels))

# Detect and recognize face
def detect_and_recognize_face(frame, target_names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        if names_dict[label] in target_names and confidence < 100:  # Adjust confidence threshold
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, names_dict[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            center_x = x + w/2
            center_y = y + h/2
            offset_x = center_x - frame.shape[1]/2
            offset_y = center_y - frame.shape[0]/2
            return (offset_x, offset_y)
    return None

# Main loop
cap = cv2.VideoCapture(0)  # Adjust the camera index
fps_control = 10  # Adjust the FPS for face detection
prev = 0

target_names = ["name1", "name2"]  # Replace with names received from other function

while True:
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if ret and time_elapsed > 1./fps_control:
        prev = time.time()
        offset = detect_and_recognize_face(frame, target_names)
        if offset:
            print(f"Face detected with offset: {offset}")

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
