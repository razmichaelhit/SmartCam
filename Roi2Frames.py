import cv2
import numpy as np
from adafruit_servokit import ServoKit

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#cv2.startWindowThread()


#display width and height
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)


#Parameters
goFlag = 0
x1 = 0 
x2 = 0
y1 = 0
y2 = 0

face_cascade = cv2.CascadeClassifier('Cascade_Detection.xml')

#define servo channels
kit=ServoKit(channels=16) 


#draw ROI - Region Of Interest 
def mouse_click(event,x,y,flags,params):
    global x1,y1,x2,y2
    global goFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        goFlag = 0
    if event == cv2.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        goFlag = 1




def nothing(x):
    pass


def create_panTilt_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.moveWindow('Trackbars',1320,0)
    cv2.createTrackbar('Pan', 'Trackbars',90,180,nothing)
    cv2.createTrackbar('Tilt', 'Trackbars',90,180,nothing)


class imxCamera:    
    def __init__(self, pan , tilt):
        self.pan = pan
        self.tilt = tilt


    def change_position(pan ,tilt):
        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt 

class FrameView:
    def __init__(self):
        self.frame_pan = 90
        self.frame_tilt = 90
        self.roi = np.zeros(4)
    
    def __str__(self):
        return'Roi: {self.roi} pan {self.frame_pan} tilt: {self.frame_tilt}'.format(self = self)

    def get_roi(self):
        self.roi[0] = x1
        self.roi[1] = x2
        self.roi[2] = y1
        self.roi[3] = y2


    def set_roi(self):
        ret, frame = cam.read()
        cv2.imshow('nanoCam',frame)
        cv2.moveWindow('nanoCam',0,0)
        cv2.setMouseCallback('nanoCam', mouse_click)
        if goFlag == 1:
            frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
            region_of_interest = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)        
            cv2.imshow('nanoCam', region_of_interest)

    def init_frame1(self):     
        while True:
            self.frame_pan = cv2.getTrackbarPos('Pan','Trackbars')
            self.frame_tilt = cv2.getTrackbarPos('Tilt', 'Trackbars') 
            imxCamera.change_position(self.frame_pan, self.frame_tilt)
            FrameView.set_roi(self)   
            if cv2.waitKey(1)==ord('1'): 
                FrameView.get_roi(self)
                print("Region Of Interest x1 ,x2, y1, y2 : " ,self.roi)
                break
            
    
    def init_right_frame(self, frame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - 60*(int(frame_idx)-1)
        print("before:", self.frame_pan)
        imxCamera.change_position(self.frame_pan, self.frame_tilt)
        print("after:" ,self.frame_pan)
        while True:
            FrameView.set_roi(self)
            if cv2.waitKey(1)==ord(frame_idx):
                FrameView.get_roi(self)   
                break
    


def detector(frame1, frame2):
    pan = frame1.frame_pan
    tilt = frame1.frame_tilt
    imxCamera.change_position(frame1.frame_pan, frame1.frame_tilt)
    while True:   
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#        faces = face_cascade.detectMultiScale(gray, 1.1, 4)   #Face Detector
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )    #Person detector
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        print(boxes)
        global x, y, w, h
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    

            objX=x+w    # X position of the right bb of the object we tracking
            objY=y+h    # Y position of the top of the object we tracking
            
            #error between the center of the frame to the object
            errorPan_right = objX-DISPLAY_WIDTH/2
            errorPan_left = x - DISPLAY_HEIGHT/2   
            errorTilt=objY-DISPLAY_HEIGHT/2 
            
            if abs(errorPan_right)>DISPLAY_WIDTH/2-5:
                pan=frame2.frame_pan
            if abs(errorPan_left)>DISPLAY_HEIGHT/2-5:
                pan=frame1.frame_pan

            if pan>180:
                pan=180
                print("Pan Out of  Range")   
            if pan<0:
                pan=0
                print("Pan Out of  Range") 
            if tilt>180:
                tilt=180
                print("Tilt Out of  Range") 
            if tilt<0:
                tilt=0
                print("Tilt Out of  Range")                 


            kit.servo[0].angle=pan
            kit.servo[1].angle=tilt 

        cv2.imshow('GrayDetectoion', frame)
        cv2.moveWindow('GrayDetectoion',530,0)

        
        if cv2.waitKey(1)==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

def main():
    create_panTilt_trackbars()
    frame1 = FrameView()
    frame1.init_frame1()
    print(frame1)
    frame2 = FrameView()
    frame2.init_right_frame('2', frame1.frame_pan, frame1.frame_tilt)   
    print(frame2)
    #cam.release()
    cv2.destroyAllWindows()
    detector(frame1, frame2)
    return

main()

