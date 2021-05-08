import cv2
import imutils
import shutil
import time
import numpy as np
import os, sys
from PIL import Image
import face_recognition
from adafruit_servokit import ServoKit
import datetime
import threading
import concurrent.futures
import smtplib
import argparse
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

#display width and height
WIDTH = 2592
HEIGHT = 1944
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FOV = 40 # Servo FOV to change frame
#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=4032, height=3040, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(WIDTH)+', height='+str(HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)

#If workinng mod = 1 , dont show video after initialize, else show video

#Parameters to mouse click event
goFlag = 0
x1 = 0
x2 = 0
y1 = 0
y2 = 0

#define servo channels
kit=ServoKit(channels=16)

def parser():
    parser = argparse.ArgumentParser(description="Smart Camera detection and classification")

    parser.add_argument("--weights", default="./cfg/Tinyyolov4_personDetection.weights",
                        help="yolo weights path")

    parser.add_argument("--cfg", default="./cfg/Tinyyolov4_personDetection.cfg",
                        help="yolo cfg path, defult: ./cfg/Tinyyolov4_personDetection.cfg")       

    parser.add_argument("--backup_dir", default="/home/rami/Desktop/SmartCam/Pictures/backup",
                        help="backup image directory")       

    parser.add_argument("--unknown_dir", default="/home/rami/Desktop/SmartCam/Pictures/unknown",
                        help="unknown image directory")       

    parser.add_argument("--known_dir", default="/home/rami/Desktop/SmartCam/Pictures/known",
                        help="known image directory")       

    parser.add_argument("--model", default="hog",
                        help="model for the classifier , cnn or hog, defullt: hog")

    parser.add_argument("--frames", type=int ,default=2,
                        help="number of frames to init, defult: 2")                                            

    parser.add_argument("--mode", type=int ,default=1,
                        help="to show video processing insert 0 , defult: 1")     

    parser.add_argument("--thresh", type=int, default=5,
                        help="false classification with confidence below this value (percentage), defult : 5")
    return parser.parse_args()



def check_arguments_errors(args):
    assert 0 < args.thresh < 100, "Threshold should be an int between zero and 100"
    assert 0 < args.frames < 4, "Frames should be an int between 1 - 3 "
    assert  args.mode == 0 or args.mode == 1, "mode should be zero or one "
    if not os.path.exists(args.cfg):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.cfg))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.backup_dir):
        raise(ValueError("Invalid backup image path {}".format(os.path.abspath(args.backup_dir))))
    if not os.path.exists(args.unknown_dir):
        raise(ValueError("Invalid unknown image path {}".format(os.path.abspath(args.unknown_dir))))
    if not os.path.exists(args.known_dir):
        raise(ValueError("Invalid known image path {}".format(os.path.abspath(args.known_dir))))

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
    cv2.createTrackbar('Pan', 'Trackbars',90,180,nothing)
    cv2.createTrackbar('Tilt', 'Trackbars',90,180,nothing)
    cv2.moveWindow('Trackbars',1320,0)



class Point:
    def __init__(self, x, y):
        self.x = x * (WIDTH/DISPLAY_WIDTH)
        self.y = y * (HEIGHT/DISPLAY_HEIGHT)



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
        frame = cv2.resize(frame,(640 , 480))
        cv2.imshow('nanoCam',frame)
        cv2.moveWindow('nanoCam',0,0)
        cv2.setMouseCallback('nanoCam', mouse_click)
        if goFlag == 1:
            frame=cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
            region_of_interest = cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0), 3)
            cv2.imshow('nanoCam', region_of_interest)

#get the roi to the frame roi
    def getRoi(self):
        self.roi[0] = x1*WIDTH/DISPLAY_WIDTH
        self.roi[1] = x2*WIDTH/DISPLAY_WIDTH
        self.roi[2] = y1*HEIGHT/DISPLAY_HEIGHT
        self.roi[3] = y2*HEIGHT/DISPLAY_HEIGHT
    

#show the roi
    def showRoi(self,frame, mode):
        frame=cv2.rectangle(frame,(int(self.roi[0]),int(self.roi[2])),(int(self.roi[1]),int(self.roi[3])),(255,0,0),3)
        region_of_interest = cv2.rectangle(frame,(int(self.roi[0]),int(self.roi[2])),(int(self.roi[1]),int(self.roi[3])),(255,0,0), 3)
        if mode == 0: cv2.imshow('nanoCam', region_of_interest)

#init the first frame
    def initFrame1(self):
        while True:
            self.frame_pan = cv2.getTrackbarPos('Pan','Trackbars')
            self.frame_tilt = cv2.getTrackbarPos('Tilt', 'Trackbars')
            if (imxCamera.changePosition(self.frame_pan, self.frame_tilt)) == 1:
                print("Failed to initialize frame:")
                print("Frame cannot be over 180 degrees or less than 0")
                return False
            FrameView.setRoi(self)
            if cv2.waitKey(1)==ord('1'):
                FrameView.getRoi(self)
                return True
            if cv2.waitKey(1)==ord('q'):
                return False
                

#init any other frame with frame_frame_idx
    def initRightFrame(self, frame_frame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - FOV*(frame_frame_idx-1)
        if (imxCamera.changePosition(self.frame_pan, self.frame_tilt)) == 1:
                print("Failed to initialize frame:")
                print("Frame cannot be over 180 degrees or less than 0")
                return False
        while True:
            FrameView.setRoi(self)
            if cv2.waitKey(1)==ord(str(frame_frame_idx)):
                FrameView.getRoi(self)
                return True
            if cv2.waitKey(1)==ord('q'):
                return False



#show the frame includes the roi
    def showFrame(self, frame, mode):
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        FrameView.showRoi(self, frame, mode)


def personDetector(frame, frame_idx, numberOfFrames, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, class_names, net, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, COLORS):
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
            print("Frame changed flag: " , FRAME_CHANGED_FLAG)
            FRAME_CHANGED_FLAG = FRAME_CHANGED_FLAG+1
            if FRAME_CHANGED_FLAG > 10:
                FRAME_CHANGED_FLAG = 0
            return frame, frame_idx, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, boundingBox
        #Checking what direction subject goes
        frame_idx, FRAME_CHANGED_FLAG = checkDirection(startX,endX,frame_idx,FRAME_CHANGED_FLAG, numberOfFrames)
        return frame, frame_idx, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, boundingBox
    #IF NO PERSON IN THE PIC FOR MORE THAN 10 FRAMES BACK TO BEGINNING
    NO_PERSON_FLAG = NO_PERSON_FLAG + 1
    if NO_PERSON_FLAG > 0:
        if NO_PERSON_FLAG > 50:
            frame_idx = 0
            NO_PERSON_FLAG = 0    
    return frame, frame_idx, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, boundingBox


def checkDirection(startX, endX, frame_idx, FRAME_CHANGED_FLAG, numberOfFrames):
    if endX>WIDTH-10:
        if frame_idx == numberOfFrames-1:
            return frame_idx, FRAME_CHANGED_FLAG
        frame_idx = frame_idx + 1
        print("Frame idx end" , frame_idx, "EndX ", endX) 
        FRAME_CHANGED_FLAG = 1
    if startX<10:
        if frame_idx == 0:
            return frame_idx, FRAME_CHANGED_FLAG
        frame_idx = frame_idx - 1
        FRAME_CHANGED_FLAG = 1
        print("Begin" , frame_idx , "StartX : " , startX)
    return frame_idx, FRAME_CHANGED_FLAG

def isInRoi(boundingBox, frame_right, frame_idx):
    (startX, startY, w, h) = boundingBox.astype("int")
    print("Frame IDX: " ,frame_idx)
    RoiBoxLeftUp = Point(frame_right[frame_idx].roi[0],frame_right[frame_idx].roi[2])
    RoiBoxLeftDown = Point(frame_right[frame_idx].roi[0],frame_right[frame_idx].roi[3])
    RoiBoxRightUp = Point(frame_right[frame_idx].roi[1],frame_right[frame_idx].roi[2])
    RoiBoxRightDown = Point(frame_right[frame_idx].roi[1],frame_right[frame_idx].roi[3])
    BoundingBoxLeftUp = Point(startX, startY)
    BoundingBoxLeftDown = Point(startX, startY + h)
    BoundingBoxRightUp = Point(startX + w, startY)
    BoundingBoxRightDown = Point(startX + w, startY + h)
    
    if(doOverlap(BoundingBoxLeftUp, BoundingBoxLeftDown, BoundingBoxRightUp, BoundingBoxRightDown, 
    RoiBoxLeftUp, RoiBoxLeftDown, RoiBoxRightUp, RoiBoxRightDown)==True):
        return True
    else:
        return False

def doOverlap(BoundingBoxLeftUp, BoundingBoxLeftDown, BoundingBoxRightUp, BoundingBoxRightDown, 
RoiBoxLeftUp, RoiBoxLeftDown, RoiBoxRightUp, RoiBoxRightDown):
    #Check if one of the bounding box points in the roi rectangle
    if (BoundingBoxLeftUp.x > RoiBoxLeftDown.x and BoundingBoxLeftUp.x < RoiBoxRightDown.x) and (BoundingBoxLeftUp.y > RoiBoxLeftUp.y and BoundingBoxLeftUp.y < RoiBoxLeftDown.y):
        return True
    if (BoundingBoxRightUp.x > RoiBoxLeftDown.x and BoundingBoxRightUp.x < RoiBoxRightDown.x) and (BoundingBoxRightUp.y > RoiBoxLeftUp.y and BoundingBoxRightUp.y < RoiBoxLeftDown.y):
        return True
    if (BoundingBoxRightDown.x > RoiBoxLeftDown.x and BoundingBoxRightDown.x < RoiBoxRightDown.x) and (BoundingBoxRightDown.y > RoiBoxLeftUp.y and BoundingBoxRightDown.y < RoiBoxLeftDown.y):
        return True
    if (BoundingBoxLeftDown.x > RoiBoxLeftDown.x and BoundingBoxLeftDown.x < RoiBoxRightDown.x) and (BoundingBoxLeftDown.y > RoiBoxLeftUp.y and BoundingBoxLeftDown.y < RoiBoxLeftDown.y):
        return True
    #Check if one of the roi box points in the bounding box rectangle
    if (RoiBoxLeftUp.x > BoundingBoxLeftDown.x and RoiBoxLeftUp.x < BoundingBoxRightDown.x) and (RoiBoxLeftUp.y > BoundingBoxLeftUp.y and RoiBoxLeftUp.y < BoundingBoxLeftDown.y):
        return True
    if (RoiBoxRightUp.x > BoundingBoxLeftDown.x and RoiBoxRightUp.x < BoundingBoxRightDown.x) and (RoiBoxRightUp.y > BoundingBoxLeftUp.y and RoiBoxRightUp.y < BoundingBoxLeftDown.y):
        return True
    if (RoiBoxRightDown.x > BoundingBoxLeftDown.x and RoiBoxRightDown.x < BoundingBoxRightDown.x) and (RoiBoxRightDown.y > BoundingBoxLeftUp.y and RoiBoxRightDown.y < BoundingBoxLeftDown.y):
        return True
    if (RoiBoxLeftDown.x > BoundingBoxLeftDown.x and RoiBoxLeftDown.x < BoundingBoxRightDown.x) and (RoiBoxLeftDown.y > BoundingBoxLeftUp.y and RoiBoxLeftDown.y < BoundingBoxLeftDown.y):
        return True
    return False


Encodings=[]
Names=[]


def classifyAndBackup(known_image_dir, unknown_image_dir, backup, frame_right, SuspectThreshhold, frame_idx, model):
    score = isSuspect(known_image_dir, unknown_image_dir, model)
    print(score)
    shutil.move(unknown_image_dir, backup)
    newBackupPath = '/home/rami/Desktop/SmartCam/Pictures/backup/'+str(datetime.datetime.now())+''
    os.rename('/home/rami/Desktop/SmartCam/Pictures/backup/unknown' , newBackupPath)
    os.mkdir(unknown_image_dir)
    pathToExamplePic = newBackupPath+'/0_unknown.png'
    if score > SuspectThreshhold:
        frame_idx = 0
        shutil.rmtree(newBackupPath)
    else:
        SendMail(pathToExamplePic)
    WriteFrameToFile(frame_idx)
    return score
    

def isSuspect(known_image_dir, unknown_image_dir, model):
    scan_known_images(known_image_dir)
    return scan_unknow_images(unknown_image_dir, model)

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
 
def scan_unknow_images(unknown_image_dir, model_net):
    vec = []
    for root,dirs, files in os.walk(unknown_image_dir):
        for file in files:
            print(root)
            print(file)
            testImagePath=os.path.join(root,file)
            testImage=face_recognition.load_image_file(testImagePath)
            facePositions=face_recognition.face_locations(testImage, model=model_net)
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

def SendMail(ImgFileName):
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    img_data = open(ImgFileName, 'rb').read()
    msg = MIMEMultipart()
    msg['Subject'] = 'Suspect in the house: '
    msg['From'] = 'razmichaelhit@gmail.com'
    msg['To'] = 'razmichaelhit@gmail.com'
    password = "Hit123456!"
    username = "razmichaelhit@gmail.com"
    sender_mail = username
    reciever_mail = "razmichaelhit@gmail.com"

    text = MIMEText("Suspect has been found in the garden")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename(ImgFileName))
    msg.attach(image)

    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login(username, password)
    server.sendmail(sender_mail, reciever_mail, msg.as_string())
    server.quit()

def WriteFrameToFile(frame_idx):
    f = open("./cfg/frame_idx.txt", "w")
    f.write(str(frame_idx))
    f.close()


def ResizeAllPicturesInFOlder(path):
    lst_imgs = [i for i in glob.glob(path+"/*.png")]
    # It creates a folder called ltl if does't exist
    if not "Resized" in os.listdir():
        os.mkdir("Resized")  
    print(lst_imgs)
    for i in lst_imgs:
        img = Image.open(i)
        img = img.resize((500, 500), Image.ANTIALIAS)
        img.save(i[:-4] +".png")
    print("Done")

def deleteAllFilesInFolder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


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

def checkThreadResults(frame_idx, lock_thread):
    f = open("./cfg/frame_idx.txt", "r")
    frame_idx_text = f.read()
    frame_idx_text = int(frame_idx_text)
    if frame_idx_text == 0:
        frame_idx = 0
        f = open("./cfg/frame_idx.txt", "w")
        f.write("NULL")
        lock_thread = 0
    if frame_idx_text > 0 and frame_idx_text < 10:
        f = open("./cfg/frame_idx.txt", "w")
        f.write("NULL")
        lock_thread = 0      
    return frame_idx, lock_thread

def init_number_of_frames(frame_right, number_of_frames_to_init):
    #init first frame
    frame_right.append(FrameView())
    if (frame_right[0].initFrame1()) == False:
        cam.release()
        cv2.destroyAllWindows()
        return frame_right, False
    print(frame_right[0])
    if number_of_frames_to_init >1:
        i = 1
        while i < number_of_frames_to_init:
            frame_right.append(FrameView())
            if (frame_right[i].initRightFrame(i+1, frame_right[0].frame_pan, frame_right[0].frame_tilt)) == False:
                cam.release()
                cv2.destroyAllWindows()
                return frame_right, False
            print(frame_right[i])
            i = i + 1
        return frame_right, True
        
#main func
def main():
    args = parser()
    check_arguments_errors(args)
    #If show video , reduce resources use by changing classification model to hog model. less accuracy 
    if args.mode == 0:
        args.model = "hog"


    create_panTilt_trackbars()

    #transverse array of frames 
    frame_right = []
    number_of_frames_to_init = 2
    #initialize frames
    frame_right, status = init_number_of_frames(frame_right, args.frames)
    if status == False:
        return False
    cv2.destroyAllWindows()

    #FRAME_CHANGED_FLAG - if frame has been changed , stop tracking the object for number of loops, keep from latency bugsglobal FRAME_CHANGED_FLAG
    FRAME_CHANGED_FLAG = 1    

    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    #Clean unknown image dir
    deleteAllFilesInFolder(args.unknown_dir)

    #Set person class
    class_names = setClassNames()
    
    #Threshhold of how many pictures true of all pictures, in percentage
    SuspectThreshhold = 5

    #configuring yolo model
    net = setDNNParams()
    frame_idx = 0
    i = 0
    
    #Flag to know how many pictures to process
    PROCESSING_FLAG = 0
    #If there is no person any more in the picture back to beginning
    NO_PERSON_FLAG = 0

    #timer to starting classify tread 
    timer = 0

    #lock thread flag
    lock_thread = 0
    while True:
        ret, frame2 = cam.read()
        ret, frame = cam.read()
        #frame = cv2.resize(frame,(640 , 480))
        frame_right[frame_idx].showFrame(frame, args.mode)
        if PROCESSING_FLAG==2 or PROCESSING_FLAG == 3:
            frame, frame_idx, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, boundingBox = personDetector(frame, frame_idx, number_of_frames_to_init, FRAME_CHANGED_FLAG, NO_PERSON_FLAG, class_names, net, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, COLORS)
            if args.mode == 0: cv2.imshow("nanoCam", frame)
            PROCESSING_FLAG = PROCESSING_FLAG + 1
            try:
                numberOfPicturesInFolder = len([name for name in os.listdir(args.unknown_dir) if os.path.isfile(os.path.join(args.unknown_dir, name))])
            except:
                continue
            #If bounding box inside the ROI then start take pictures
            if isInRoi(boundingBox, frame_right, frame_idx):
                if  numberOfPicturesInFolder < 10 and i <10:
                    cv2.imwrite(args.unknown_dir+'/'+str(i)+'_unknown.png',frame2)
                    i = i + 1
                #If there are 30 pics in the folder start classify thread 
                if i == 10 and lock_thread == 0:
                    classify_thread = threading.Thread(target = classifyAndBackup, args=(args.known_dir, args.unknown_dir, args.backup_dir, frame_right, SuspectThreshhold, frame_idx, args.model))
                    classify_thread.start()
                    lock_thread = 1
                    i = 0
            #if timer over 100 frames and there is no processing thread on start classify thread
            if timer > 100 and lock_thread == 0:
                classify_thread = threading.Thread(target = classifyAndBackup, args=(args.known_dir, args.unknown_dir, args.backup_dir, frame_right, SuspectThreshhold, frame_idx, args.model))
                classify_thread.start()
                lock_thread = 1
                timer = 0
            #Start timer if there are any pics left in the unknown folder
            if numberOfPicturesInFolder > 0:
                timer = timer + 1
                print("timer: "  ,timer)
            if numberOfPicturesInFolder == 0:
                timer = 0
            print("lock_thread: ",lock_thread)
            
            #if no suspect return to frame 1
            try:
                frame_idx, lock_thread = checkThreadResults(frame_idx, lock_thread)
            except:
                nothing(1)

        if PROCESSING_FLAG==4:
            PROCESSING_FLAG = 0
        else:
            if args.mode == 0: cv2.imshow("nanoCam", frame)
            # if the `q` key was pressed, break from the loop
            PROCESSING_FLAG = PROCESSING_FLAG + 1
            if cv2.waitKey(1) == ord("q"):
                break


    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
