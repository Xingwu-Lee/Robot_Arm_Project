import cv2
import os
import numpy as np


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    label_map = {}

    for label, person_name in enumerate(os.listdir(data_folder_path)):
        person_dir_path = os.path.join(data_folder_path, person_name)
        person_images = os.listdir(person_dir_path)

        label_map[label] = person_name

        for image_name in person_images:
            image_path = os.path.join(person_dir_path, image_name)
            image = cv2.imread(image_path)

            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)
            else:
                print(f"Face not detected in image: {image_path}")

    return faces, np.array(labels), label_map


# 使用函数
data_folder_path = "face_data"
faces, labels, label_map = prepare_training_data(data_folder_path)

# 训练模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# 保存模型
recognizer.write("face_recognizer.yml")
np.save("label_map.npy", label_map)
