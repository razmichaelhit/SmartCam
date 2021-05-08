import numpy as np
import cv2 
from adafruit_servokit import ServoKit

print(cv2.__version__)
dispW=1920
dispH=1080
#Uncomment These next Two Line for Pi Camera
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
name= 'Raz'
#camSet='nvarguscamerasrc sensor-id=-0 ! video/x-raw(memory:NVMM), width=3264, height=2464, framerate=21/1, format=NV12 ! nvvidconv flip-method=2 ! video/x-raw, width=800, height=600, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)
#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
#cam=cv2.VideoCaptur%e(0)
i=1
kit=ServoKit(channels=16)

def nothing(x):
    pass

def create_panTilt_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('Pan', 'Trackbars',90,180,nothing)
    cv2.createTrackbar('Tilt', 'Trackbars',90,180,nothing)
    cv2.moveWindow('Trackbars',1320,0)

create_panTilt_trackbars()

#Class camera for change position
class imxCamera:
    def __init__(self, pan , tilt):
        self.pan = pan
        self.tilt = tilt


    def changePosition(pan ,tilt):
        if pan > 180 or tilt > 180 or pan<0 or tilt<0:
            print("pan or tilt cannot be more than 180")
            return 1
        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt


while True:
    pan = cv2.getTrackbarPos('Pan','Trackbars')
    tilt = cv2.getTrackbarPos('Tilt', 'Trackbars')
    ret, frame = cam.read()
    imxCamera.changePosition(pan,tilt)
    cv2.imshow('nanoCam',frame)

    if cv2.waitKey(1)==ord('p'):
        cv2.imwrite('/home/rami/Desktop/SmartCam/Pictures/known/'+str(i)+name+'.png',frame)
        i = i + 1
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
