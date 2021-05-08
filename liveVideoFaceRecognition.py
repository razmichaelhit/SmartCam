import face_recognition
import cv2
import os
import pickle
print(cv2.__version__)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
Encodings = []
Names = []
 

with open('train.pkl', 'rb') as f:
    Names=pickle.load(f)
    Encodings=pickle.load(f)


camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! nvvidconv  ! video/x-raw, width='+str(DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cam.read()
    frameSmall = cv2.resize(frame,(0,0) , fx=.33 , fy=.33)
    frameRGB = cv2.cvtColor(frameSmall, cv2.COLOR_BGR2RGB)
    facePositions=face_recognition.face_locations(frameRGB, model='cnn')
    allEncodings=face_recognition.face_encodings(frameRGB,facePositions)
    for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
        name='Unkowm Person'
        matches=face_recognition.compare_faces(Encodings,face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = Names[first_match_index]
        top = top*3
        right = right*3
        bottom = bottom*3
        left = left*3
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-6),font,.75,(0,0,255),2)
    cv2.imshow('Picture',frame)
    cv2.moveWindow('Picture', 0 ,0)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows