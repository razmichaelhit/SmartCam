import cv2
import imutils
import time
import numpy as np
from adafruit_servokit import ServoKit



#display width and height
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
FOV = 40 # Servo FOV to change frame
#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)

#Parameters to mouse click event
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
        if pan > 180 or tilt > 180 or pan<0 or tilt<0:
            print("pan or tilt cannot be more than 180")
            return 1
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

#get the roi to the frame roi
    def getRoi(self):
        self.roi[0] = x1
        self.roi[1] = x2
        self.roi[2] = y1
        self.roi[3] = y2


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
            if cv2.waitKey(1)==ord('q'):
                break

#init any other frame with frame_fame_idx
    def initRightFrame(self, frame_fame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - FOV*(int(frame_fame_idx)-1)
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        while True:
            FrameView.setRoi(self)
            if cv2.waitKey(1)==ord(frame_fame_idx):
                FrameView.getRoi(self)
                break
            if cv2.waitKey(1)==ord('q'):
               break


#show the frame includes the roi
    def showFrame(self, frame):
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        FrameView.showRoi(self, frame)
        #is_suspect()


#face detection Deep Nueral Network for first tracking object
def faceDetector(frame_right):
    
    #FRAME_CHANGED_FLAG - if frame has been changed , stop tracking the object for number of loops, keep from latency bugsglobal FRAME_CHANGED_FLAG
    FRAME_CHANGED_FLAG = 0
    
    PROCESSING_FLAG = 0 #to reduce latency we processing 2 of every 4 frames
    fame_idx = 0
    #Loading DNN Network to object detection
    prototext = "deploy.prototxt"
    model = "person_detection_0022.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototext, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    while True:
        PROCESSING_FLAG = PROCESSING_FLAG + 1
        ret, frame = cam.read()
        #change shape for dnn
        (h, w) = frame.shape[:2]
        #Showing the frame by the frame indx
        frame_right[fame_idx].showFrame(frame)
        if PROCESSING_FLAG==2 or PROCESSING_FLAG==3:
            
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                #if the detection accuracy lower the 30% continue
                confidence = detections[0, 0, i, 2]
                if confidence < 0.3:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                
                #Draw Accuracy Percentages
                cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
               
                #Tracking after the subject with the camera
                if FRAME_CHANGED_FLAG>0:
                    FRAME_CHANGED_FLAG = FRAME_CHANGED_FLAG+1
                    if FRAME_CHANGED_FLAG > 10:
                        FRAME_CHANGED_FLAG = 0
                    continue
                
                if endX>DISPLAY_WIDTH-2:
                    fame_idx = fame_idx + 1
                    FRAME_CHANGED_FLAG = 1
                    print(fame_idx)
                    continue
                if fame_idx == 0:
                    continue
                if startX<2:
                    fame_idx = fame_idx - 1
                    FRAME_CHANGED_FLAG = 1
                    continue

            cv2.imshow("nanoCam", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        if PROCESSING_FLAG==4:
            PROCESSING_FLAG = 0
        else:
            cv2.imshow("nanoCam", frame)
            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) == ord("q"):
                break
    cam.release()
    cv2.destroyAllWindows()

        


#main func
def main():
    create_panTilt_trackbars()
    
    #transverse array of frames 
    frame_right = []
    
    frame_right.append(FrameView())
    frame_right[0].initFrame1()
    print(frame_right[0])

    frame_right.append(FrameView())
    frame_right[1].initRightFrame('2', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[1])
    
    frame_right.append(FrameView())
    frame_right[2].initRightFrame('3', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[2])

    #detct Faces
    faceDetector(frame_right)
    return

main()
