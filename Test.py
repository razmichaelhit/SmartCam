import numpy as np
import cv2 
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
while True:
    ret, frame = cam.read()
    
    cv2.imshow('nanoCam',frame)

    if cv2.waitKey(1)==ord('p'):
        cv2.imwrite('/home/rami/Desktop/Project/Pictures/known/'+str(i)+name+'.png',frame)
        i = i + 1
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
