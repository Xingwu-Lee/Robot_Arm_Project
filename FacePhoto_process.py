import cv2
import os
import uuid
import time

# 加载Haar级联分类器用于人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# 定义一个函数来检测并处理单张人脸照片
def process_single_face(image_path, output_directory, output_size=(100, 100)):
    # 读取图像
    img = cv2.imread(image_path)

    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        # 裁剪人脸部分
        face_roi = gray[y:y + h, x:x + w]

        # 调整图像大小为统一尺寸
        face_resized = cv2.resize(face_roi, output_size)

        # 生成唯一的输出文件名（使用UUID）
        unique_filename = str(uuid.uuid4())[:8] + '.jpg'
        output_path = os.path.join(output_directory, unique_filename)

        # 保存统一尺寸的人脸图像
        cv2.imwrite(output_path, face_resized)


# 处理整个目录中的所有人脸照片
def process_images_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_directory, filename)
            process_single_face(image_path, output_directory)


if __name__ == "__main__":
    input_directory = 'input_images'  # 存放人脸照片的目录
    output_directory = 'output_faces'  # 存放统一尺寸人脸图像的目录
    process_images_in_directory(input_directory, output_directory)
