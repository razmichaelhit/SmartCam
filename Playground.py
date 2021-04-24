import cv2
import imutils
import shutil
import time
import numpy as np
import os
import face_recognition
from adafruit_servokit import ServoKit
import datetime
import threading
import concurrent.futures

#display width and height
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
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


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

#init any other frame with frame_frame_idx
    def initRightFrame(self, frame_frame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - FOV*(int(frame_frame_idx)-1)
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        while True:
            FrameView.setRoi(self)
            if cv2.waitKey(1)==ord(frame_frame_idx):
                FrameView.getRoi(self)
                break
            if cv2.waitKey(1)==ord('q'):
               break


#show the frame includes the roi
    def showFrame(self, frame):
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        FrameView.showRoi(self, frame)


def personDetector(frame, frame_idx, FRAME_CHANGED_FLAG, class_names, net, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, COLORS):
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    boundingBox = np.zeros(4)
    for (classid, score, boundingBox) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        (startX, startY, w, h) = boundingBox.astype("int")
        endX = startX + w
        endY = startY + h
        cv2.rectangle(frame, boundingBox, color, 2)
        cv2.putText(frame, label, (boundingBox[0], boundingBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #Dealing with latency between servo and frame 
        if FRAME_CHANGED_FLAG>0:
            FRAME_CHANGED_FLAG = FRAME_CHANGED_FLAG+1
            if FRAME_CHANGED_FLAG > 10:
                FRAME_CHANGED_FLAG = 0
            return frame, frame_idx, FRAME_CHANGED_FLAG, boundingBox

        #Checking what direction subject goes
        frame_idx, FRAME_CHANGED_FLAG = checkDirection(startX,endX,frame_idx,FRAME_CHANGED_FLAG)
    return frame, frame_idx, FRAME_CHANGED_FLAG, boundingBox


def checkDirection(startX, endX, frame_idx, FRAME_CHANGED_FLAG):
    if frame_idx < 0:
        frame_idx = 0
        return frame_idx, FRAME_CHANGED_FLAG
    if endX>DISPLAY_WIDTH-10:
        frame_idx = frame_idx + 1
        FRAME_CHANGED_FLAG = 1
    if startX<10:
        frame_idx = frame_idx - 1
        FRAME_CHANGED_FLAG = 1
    return frame_idx, FRAME_CHANGED_FLAG

def isInRoi(boundingBox, frame_right, frame_idx):
    (startX, startY, w, h) = boundingBox.astype("int")
    l1 = Point(frame_right[frame_idx].roi[0],frame_right[frame_idx].roi[2])
    r1 = Point(frame_right[frame_idx].roi[1],frame_right[frame_idx].roi[3])
    l2 = Point(startX, startY)
    r2 = Point(startX + w,  startY + h)
    if(doOverlap(l1,r1,l2,r2)==True):
        return True
    else:
        return False

def doOverlap(l1, r1, l2, r2):   
    # To check if either rectangle is actually a line
    # print("l1:", l1.x, l1.y)
    # print("r1:", r1.x, r1.y)
    # print("l2:", l2.x, l2.y)
    # print("r2:", r2.x, r2.y)
    if (l1.x > l2.x and l1.x < r2.x) and (l1.y > l2.y and l1.y < r2.y):
        return True
    if (r1.x > l2.x and r1.x < r2.x) and (r1.y > l2.y and r1.y < r2.y):
        return True
    return False


Encodings=[]
Names=[]


def classifyAndBackup(known_image_dir, unknown_image_dir, backup, frame_right, SuspectThreshhold, frame_idx):
    score = isSuspect(known_image_dir, unknown_image_dir)
    print(score)
    shutil.move(unknown_image_dir, backup)
    os.rename('/home/rami/Desktop/Project/Pictures/backup/unknown' , '/home/rami/Desktop/Project/Pictures/backup/'+str(datetime.datetime.now())+'')
    os.mkdir(unknown_image_dir)
    IS_SUSPECT_FLAG = 0
    if score > SuspectThreshhold:
        frame_idx = 0
    f = open("./cfg/frame_idx.txt", "w")
    f.write(str(frame_idx))
    f.close()
    return IS_SUSPECT_FLAG
    

def isSuspect(known_image_dir, unknown_image_dir):
    scan_known_images(known_image_dir)
    return scan_unknow_images(unknown_image_dir)

def scan_known_images(known_image_dir):
    for root, dirs, files in os.walk(known_image_dir):
        print(files)
        for file in files:
            path=os.path.join(root,file)
            print(path)
            name=os.path.splitext(file)[0]
            print(name)
            person=face_recognition.load_image_file(path)
            encoding=face_recognition.face_encodings(person)[0]
            Encodings.append(encoding)
            Names.append(name)
    print(Names)
    
font=cv2.FONT_HERSHEY_SIMPLEX
 
def scan_unknow_images(unknown_image_dir):
    vec = []
    for root,dirs, files in os.walk(unknown_image_dir):
        for file in files:
            print(root)
            print(file)
            testImagePath=os.path.join(root,file)
            testImage=face_recognition.load_image_file(testImagePath)
            facePositions=face_recognition.face_locations(testImage)
            allEncodings=face_recognition.face_encodings(testImage,facePositions)
            testImage=cv2.cvtColor(testImage,cv2.COLOR_RGB2BGR)
    
            for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
                name='Known'
                matches=face_recognition.compare_faces(Encodings,face_encoding)
                if True in matches:
                    first_match_index=matches.index(True)
                    name=Names[first_match_index]
                    print("True match")
                    vec.append(1)
            vec.append(0)
        #cv2.imshow('Picture', testImage)
        #cv2.moveWindow('Picture',0,0)
        print(vec)
        confidence = calculate_statistic(vec)
        print("Scan Unknown Images" , confidence)
        return confidence
        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()

def calculate_statistic(vec):
    len_vec = len(vec) # 0 - number of pictures   1 - true result 
    true_results = countOccurrences(vec,len_vec,1)
    number_of_pictures = countOccurrences(vec,len_vec,0)
    confidence = true_results/number_of_pictures
    confidence = confidence * 100 #For Pecentage
    print("calculate_statistic" , confidence)
    return confidence
    
def countOccurrences(arr, n, x):
    res = 0
    for i in range(n):
        if x == arr[i]:
            res += 1
    return res
  
def setDNNParams():
    #configuring yolo model
    net = cv2.dnn.readNet("./cfg/Tinyyolov4_personDetection.weights", "./cfg/Tinyyolov4_personDetection.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net

def setClassNames():
    #open class txt file (Person)
    class_names = []
    with open("./cfg/classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    return class_names

#main func
def main():
    create_panTilt_trackbars()

    #transverse array of frames 
    frame_right = []
    
    #initialize frames
    frame_right.append(FrameView())
    frame_right[0].initFrame1()
    print(frame_right[0])

    frame_right.append(FrameView())
    frame_right[1].initRightFrame('2', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[1])
    
    frame_right.append(FrameView())
    frame_right[2].initRightFrame('3', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[2])

    #FRAME_CHANGED_FLAG - if frame has been changed , stop tracking the object for number of loops, keep from latency bugsglobal FRAME_CHANGED_FLAG
    FRAME_CHANGED_FLAG = 1    
    #If Suspect
    global IS_SUSPECT_FLAG
    IS_SUSPECT_FLAG = 0

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    #Directories to save pictures to processing and suspects    
    backup = "/home/rami/Desktop/Project/Pictures/backup"
    unknown_image_dir='/home/rami/Desktop/Project/Pictures/unknown'
    known_image_dir='/home/rami/Desktop/Project/Pictures/known'

    #Set person class
    class_names = setClassNames()
    
    SuspectThreshhold = 5

    #configuring yolo model
    net = setDNNParams()
    frame_idx = 0
    i = 0
    PROCESSING_FLAG = 0
    while True:
        ret, frame = cam.read()
        frame_right[frame_idx].showFrame(frame)
        if PROCESSING_FLAG==2 or PROCESSING_FLAG == 3:
            frame, frame_idx, FRAME_CHANGED_FLAG, boundingBox = personDetector(frame, frame_idx, FRAME_CHANGED_FLAG, class_names, net, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, COLORS)
            cv2.imshow("nanoCam", frame)
            PROCESSING_FLAG = PROCESSING_FLAG + 1
            try:
                numberOfPicturesInFolder = len([name for name in os.listdir(unknown_image_dir) if os.path.isfile(os.path.join(unknown_image_dir, name))])
            except:
                continue
            if isInRoi(boundingBox, frame_right, frame_idx):
                if  numberOfPicturesInFolder < 30 and i < 30:
                    cv2.imwrite(unknown_image_dir+'/'+str(i)+'_unknown.png',frame)
                    i = i + 1
                if i == 30:
                    classify_thread = threading.Thread(target = classifyAndBackup, args=(known_image_dir, unknown_image_dir, backup, frame_right, SuspectThreshhold, frame_idx))
                    classify_thread.start()
                    i = 0
        
            try:
                f = open("./cfg/frame_idx.txt", "r")
                frame_idx_text = f.read()
                frame_idx_text = int(frame_idx_text)
                if frame_idx_text == 0:
                    frame_idx = 0
                    f = open("./cfg/frame_idx.txt", "w")
                    f.write("NULL")
            except:
                print("nothing")

        if PROCESSING_FLAG==4:
            PROCESSING_FLAG = 0
        else:
            cv2.imshow("nanoCam", frame)
            # if the `q` key was pressed, break from the loop
            PROCESSING_FLAG = PROCESSING_FLAG + 1
            if cv2.waitKey(1) == ord("q"):
                break


    cam.release()
    cv2.destroyAllWindows()


main()
