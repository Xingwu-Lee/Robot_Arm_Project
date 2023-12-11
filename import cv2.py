import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained MobileNet SSD model for face detection
net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

# Load pre-trained face recognition model (e.g., FaceNet, OpenFace)
# Assuming you have a function to load the model and get embeddings
face_rec_model = load_face_recognition_model('face_recognition_model_path')

# Function to get face embeddings
def get_face_embeddings(face_image):
    # Preprocess and pass the face image through the face recognition model
    # Return the embeddings
    pass

# Function to compare embeddings
def compare_embeddings(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])

# Load and preprocess stored faces
stored_faces = []
for filename in os.listdir('path_to_faces_folder'):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join('path_to_faces_folder', filename))
        embedding = get_face_embeddings(image)
        stored_faces.append(embedding)

# Function to detect and recognize face
def detect_and_recognize_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Adjust confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            face = frame[y:y2, x:x2]
            embedding = get_face_embeddings(face)

            # Compare with stored faces
            for stored_embedding in stored_faces:
                similarity = compare_embeddings(embedding, stored_embedding)
                if similarity > threshold:  # Set a suitable threshold
                    # Face recognized, calculate offset
                    center_x = (x + x2) / 2
                    center_y = (y + y2) / 2
                    offset_x = center_x - w / 2
                    offset_y = center_y - h / 2
                    return (offset_x, offset_y)

    return None

# Main loop
cap = cv2.VideoCapture(0)  # Adjust the camera index as needed

while True:
    ret, frame = cap.read()
    if ret:
        offset = detect_and_recognize_face(frame)
        if offset:
            # Use offset for robot arm movement
            print(f"Face detected with offset: {offset}")
        
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()
