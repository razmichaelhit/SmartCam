import cv2
import imutils
import shutil
import time
import numpy as np
import os
import face_recognition
from adafruit_servokit import ServoKit


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
        #is_suspect()


#face detection Deep Nueral Network for first tracking object
def personDetector(frame_right):
    
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    #FRAME_CHANGED_FLAG - if frame has been changed , stop tracking the object for number of loops, keep from latency bugsglobal FRAME_CHANGED_FLAG
    FRAME_CHANGED_FLAG = 0
    
    PROCESSING_FLAG = 0 #to reduce latency we processing 2 of every 4 frames
    frame_idx = 0
    i = 0

    #open class txt file (Person)
    class_names = []
    with open("classes.txt", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    #configuring yolo model
    net = cv2.dnn.readNet("Tinyyolov4_personDetection.weights", "Tinyyolov4_personDetection.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    while True:
        ret, frame = cam.read()
        #Showing the frame by the frame indx
        frame_right[frame_idx].showFrame(frame)
        
        if PROCESSING_FLAG==2 or PROCESSING_FLAG == 3:
            model = cv2.dnn_DetectionModel(net)
            model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
            classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid[0]], score)
                (startX, startY, w, h) = box.astype("int")
                endX = startX + w
                endY = startY + h
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #Tracking after the subject with the camera
                if FRAME_CHANGED_FLAG>0:
                    FRAME_CHANGED_FLAG = FRAME_CHANGED_FLAG+1
                    if FRAME_CHANGED_FLAG > 10:
                        FRAME_CHANGED_FLAG = 0
                    continue
                
                #Check if the object overlap in the frame roi
                if isInRoi(box, frame_right, frame_idx) == 1 and i < 30:
                    cv2.imwrite('/home/rami/Desktop/Project/Pictures/unknown/'+str(i)+'_unknown.png',frame)
                    i = i + 1
                if i == 15:
                    score = isSuspect()
                    print("The score is : " , score)
                    # if score<0.2:
                    #     sendMail()  
                if i == 30:
                    backup = "/home/rami/Desktop/Project/Pictures/backup"
                    unknown_image_dir='/home/rami/Desktop/Project/Pictures/unknown'
                    shutil.move(unknown_image_dir, backup)
                    os.rename('/home/rami/Desktop/Project/Pictures/backup/unknown' , '/home/rami/Desktop/Project/Pictures/backup/'+str(time.time())+'')
                    os.mkdir(unknown_image_dir)
                if i==999:
                    i = 0
                    
                #Checking what direction subject goes
                if endX>DISPLAY_WIDTH-10:
                    frame_idx = frame_idx + 1
                    FRAME_CHANGED_FLAG = 1
                    print(frame_idx)
                    continue
                if frame_idx == 0:
                    continue
                if startX<10:
                    frame_idx = frame_idx - 1
                    FRAME_CHANGED_FLAG = 1
                    continue

            cv2.imshow("nanoCam", frame)
            PROCESSING_FLAG = PROCESSING_FLAG + 1
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

def isInRoi(box, frame_right, frame_idx):
    (startX, startY, w, h) = box.astype("int")
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

def isSuspect():
    scan_known_images()
    return scan_unknow_images()

def scan_known_images():
    known_image_dir='/home/rami/Desktop/Project/Pictures/known'
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
 
def scan_unknow_images():
    unknown_image_dir='/home/rami/Desktop/Project/Pictures/unknown'
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
        print("ScanUnknowImegase" , confidence)
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
    personDetector(frame_right)
    return

main()
