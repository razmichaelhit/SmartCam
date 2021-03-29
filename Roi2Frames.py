import cv2
import imutils
import numpy as np
from adafruit_servokit import ServoKit

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

#Create Pan Tilt trackbars for first initialization
def create_panTilt_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.moveWindow('Trackbars',1320,0)
    cv2.createTrackbar('Pan', 'Trackbars',90,180,nothing)
    cv2.createTrackbar('Tilt', 'Trackbars',90,180,nothing)

#Class camera for change position
class imxCamera:    
    def __init__(self, pan , tilt):
        self.pan = pan
        self.tilt = tilt


    def changePosition(pan ,tilt):
        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt 

#Class camera for each frame of the session
class FrameView:
    def __init__(self):
        self.frame_pan = 90
        self.frame_tilt = 90
        self.roi = np.zeros(4)
    
    def __str__(self):
        return'Roi: {self.roi} pan {self.frame_pan} tilt: {self.frame_tilt}'.format(self = self)

#get the roi to the frame roi
    def getRoi(self):
        self.roi[0] = x1
        self.roi[1] = x2
        self.roi[2] = y1
        self.roi[3] = y2

#Set the roi from the user
    def setRoi(self):
        ret, frame = cam.read()
        cv2.imshow('nanoCam',frame)
        cv2.moveWindow('nanoCam',0,0)
        cv2.setMouseCallback('nanoCam', mouse_click)
        if goFlag == 1:
            frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
            region_of_interest = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)        
            cv2.imshow('nanoCam', region_of_interest)

#show the roi 
    def showRoi(self,frame):
        frame=cv2.rectangle(frame,(int(self.roi[0]),int(self.roi[2])),(int(self.roi[1]),int(self.roi[3])),(255,0,0),3)
        region_of_interest = cv2.rectangle(frame,(int(self.roi[0]),int(self.roi[2])),(int(self.roi[1]),int(self.roi[3])),(255,0,0), 3)        
        cv2.imshow('nanoCam', region_of_interest)

#init the first frame
    def initFrame1(self):     
        while True:
            self.frame_pan = cv2.getTrackbarPos('Pan','Trackbars')
            self.frame_tilt = cv2.getTrackbarPos('Tilt', 'Trackbars') 
            imxCamera.changePosition(self.frame_pan, self.frame_tilt)
            FrameView.setRoi(self)   
            if cv2.waitKey(1)==ord('1'): 
                FrameView.getRoi(self)
                break
            
#init any other frame with frame_idx
    def initRightFrame(self, frame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - 60*(int(frame_idx)-1)
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        while True:
            FrameView.setRoi(self)
            if cv2.waitKey(1)==ord(frame_idx):
                FrameView.getRoi(self)   
                break

#show the frame includes the roi
    def showFrame(self, frame):
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        FrameView.showRoi(self, frame)
        #is_suspect()
    

#face detection Deep Nueral Network for first tracking object
def faceDetector(frame1, frame2):
    idx = 1
    prototext = "deploy_lowers.prototxt"
    model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototext, model)
    while True:
        ret, frame = cam.read()
        (h, w) = frame.shape[:2]
        if(idx == 1):
            frame1.showFrame(frame)

        if(idx == 2):
            frame2.showFrame(frame)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
        net.setInput(blob) 
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.3:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            #error between the center of the frame to the object
            errorPan_right = endX
            errorPan_left = startX - DISPLAY_WIDTH
            if abs(errorPan_right)>DISPLAY_WIDTH-5:
                idx = 2
            if abs(errorPan_left)>DISPLAY_WIDTH-5:
                idx = 1

        # show the output frame
        cv2.imshow("nanoCam", frame)        
    
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()



#main func
def main():
    create_panTilt_trackbars()
    frame1 = FrameView()
    frame1.initFrame1()
    print(frame1)
    frame2 = FrameView()
    frame2.initRightFrame('2', frame1.frame_pan, frame1.frame_tilt)   
    print(frame2)
    cv2.destroyAllWindows()
    faceDetector(frame1, frame2)
    return

main()
####################################################### har cascade detector - bad ############################################
# def detector(frame1, frame2):
#     pan = frame1.frame_pan
#     tilt = frame1.frame_tilt
#     imxCamera.changePosition(frame1.frame_pan, frame1.frame_tilt)
#     while True:   
#         ret, frame = cam.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #        faces = face_cascade.detectMultiScale(gray, 1.1, 4)   #Face Detector
#         boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )    #Person detector
#         boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
#         print(boxes)
#         global x, y, w, h
#         for (x, y, w, h) in boxes:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    

#             objX=x+w    # X position of the right bb of the object we tracking
#             objY=y+h    # Y position of the top of the object we tracking
            
#             #error between the center of the frame to the object
#             errorPan_right = objX-DISPLAY_WIDTH/2
#             errorPan_left = x - DISPLAY_HEIGHT/2   
#             errorTilt=objY-DISPLAY_HEIGHT/2 
            
#             if abs(errorPan_right)>DISPLAY_WIDTH/2-5:
#                 pan=frame2.frame_pan
#             if abs(errorPan_left)>DISPLAY_HEIGHT/2-5:
#                 pan=frame1.frame_pan

#             if pan>180:
#                 pan=180
#                 print("Pan Out of  Range")   
#             if pan<0:
#                 pan=0
#                 print("Pan Out of  Range") 
#             if tilt>180:
#                 tilt=180
#                 print("Tilt Out of  Range") 
#             if tilt<0:
#                 tilt=0
#                 print("Tilt Out of  Range")                 


#             kit.servo[0].angle=pan
#             kit.servo[1].angle=tilt 

#         cv2.imshow('GrayDetectoion', frame)
#         cv2.moveWindow('GrayDetectoion',530,0)

        
#         if cv2.waitKey(1)==ord('q'):
#             break
#     cam.release()
#     cv2.destroyAllWindows()