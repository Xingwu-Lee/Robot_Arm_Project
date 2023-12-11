#!/usr/bin/python3
# coding=utf8
import sys
import cv2
import math
import time
import threading
import numpy as np
import HiwonderSDK.Board as Board
import HiwonderSDK.PID as PID
import HiwonderSDK.Misc as Misc
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import face_recognition
import os
import glob

def load_face_encodings(name):
    """
    Load face encodings for the given name from a specific folder structure.
    """
    known_face_encodings = []
    folder_path = os.path.join("output_faces", name)  # "faces" is the main folder containing subfolders for each individual

    # Load all images from the subfolder
    for image_path in glob.glob(os.path.join(folder_path, "*.jpg")):
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)

    return known_face_encodings


# Ask for the name
name_to_track = input("Enter the name of the person to track: ")

# Load encodings for the given name
known_face_encodings = load_face_encodings(name_to_track)





# 人脸检测
if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

AK = ArmIK()

x_dis = 500
y_dis = 10
Z_DIS = 18
z_dis = Z_DIS
x_pid = PID.PID(P=0.15, I=0.00, D=0.01)  # pid初始化
y_pid = PID.PID(P=0.00001, I=0, D=0)
z_pid = PID.PID(P=0.005, I=0, D=0)
    
# 初始位置
def run(img):
    global x_dis, y_dis, z_dis, st

    # Convert the image from BGR color to RGB color
    rgb_frame = img[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if True in matches:
            # Save the image if it's the person we want
            face_image = img[top:bottom, left:right]
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"tracked_faces/{name_to_track}_{timestamp}.jpg", face_image)

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(img, name_to_track, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            # Face tracking logic
            X = (left + right) / 2
            Y = (top + bottom) / 2

            x_pid.SetPoint = img.shape[1] / 2.0  # Desired X position (center)
            x_pid.update(X)  # Current X position
            dx = x_pid.output
            x_dis += int(dx)  # Update X displacement

            x_dis = max(0, min(x_dis, 1000))  # Constrain X displacement

            if abs(Y - img.shape[0] / 2.0) < 20:
                z_pid.SetPoint = Y
            else:
                z_pid.SetPoint = img.shape[0] / 2.0  # Desired Y position (center)

            z_pid.update(Y)
            dy = z_pid.output
            z_dis += dy  # Update Z displacement

            z_dis = max(10.00, min(z_dis, 40.00))  # Constrain Z displacement

            target = AK.setPitchRange((0, round(y_dis, 2), round(z_dis, 2)), -5, 10)

            if target:
                servo_data = target[0]
                if st:
                    Board.setBusServoPulse(3, servo_data['servo3'], 1000)
                    Board.setBusServoPulse(4, servo_data['servo4'], 1000)
                    Board.setBusServoPulse(5, servo_data['servo5'], 1000)
                    time.sleep(1)
                    st = False
                else:
                    Board.setBusServoPulse(3, servo_data['servo3'], 20)
                    Board.setBusServoPulse(4, servo_data['servo4'], 20)
                    Board.setBusServoPulse(5, servo_data['servo5'], 20)

            Board.setBusServoPulse(6, int(x_dis), 20)
            time.sleep(0.03)

    return img
# 阈值
conf_threshold = 0.6
# 模型位置
modelFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "./models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

frame_pass = True
x1=x2=y1=y2 = 0
old_time = 0
st = True

def run(img):
    global st 
    global old_time
    global frame_pass
    global x1,x2,y1,y2
    global x_dis, y_dis, z_dis
    
    if not frame_pass:
        frame_pass = True
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
        x1=x2=y1=y2 = 0
        return img
    else:
        frame_pass = False
        
    img_copy = img.copy()
    img_h, img_w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img_copy, 1, (150, 150), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward() #计算识别
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            #识别到的人了的各个坐标转换会未缩放前的坐标
            x1 = int(detections[0, 0, i, 3] * img_w)
            y1 = int(detections[0, 0, i, 4] * img_h)
            x2 = int(detections[0, 0, i, 5] * img_w)
            y2 = int(detections[0, 0, i, 6] * img_h)             
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2, 8) #将识别到的人脸框出
##area_max = int(abs((x2-x1)*(y2-y1)))
            
            X = (x1 + x2)/2
            Y = (y1 + y2)/2
            
            x_pid.SetPoint = img_w / 2.0  # 设定
            x_pid.update(X)  # 当前
            dx = x_pid.output
            x_dis += int(dx)  # 输出

            x_dis = 0 if x_dis < 0 else x_dis
            x_dis = 1000 if x_dis > 1000 else x_dis

            
            if abs(Y - img_h/2.0) < 20:
                z_pid.SetPoint = Y
            else:
                z_pid.SetPoint = img_h / 2.0
                
            z_pid.update(Y)
            dy = z_pid.output
            z_dis += dy

            z_dis = 40.00 if z_dis > 40.00 else z_dis
            z_dis = 10.00 if z_dis < 8.00 else z_dis
            
            target = AK.setPitchRange((0, round(y_dis, 2), round(z_dis, 2)), -5, 10)
            
            if target:
                servo_data = target[0]
                if st:
                    Board.setBusServoPulse(3, servo_data['servo3'], 1000)
                    Board.setBusServoPulse(4, servo_data['servo4'], 1000)
                    Board.setBusServoPulse(5, servo_data['servo5'], 1000)
                    time.sleep(1)
                    st = False
                else:
                    Board.setBusServoPulse(3, servo_data['servo3'], 20)
                    Board.setBusServoPulse(4, servo_data['servo4'], 20)
                    Board.setBusServoPulse(5, servo_data['servo5'], 20)
                    
            Board.setBusServoPulse(6, int(x_dis), 20)
            time.sleep(0.03)
       
    return img

if __name__ == '__main__':
    initMove()
    cap = cv2.VideoCapture(-1) #读取摄像头
    
    while True:
        ret, img = cap.read()
        if ret:
            frame = img.copy()
            Frame = run(frame)           
            frame_resize = cv2.resize(Frame, (320, 240))
            cv2.imshow('frame', frame_resize)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)
    cv2.destroyAllWindows()


        

