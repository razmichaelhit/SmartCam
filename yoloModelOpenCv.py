import cv2
import time


#display width and height
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FOV = 40 # Servo FOV to change frame
#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1080, height=1920, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'


CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


vc= cv2.VideoCapture(camSet)

net = cv2.dnn.readNet("Tinyyolov4_personDetection.weights", "Tinyyolov4_personDetection.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

PROCESSINGFLAG= 0
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if PROCESSINGFLAG == 1:
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            (startX, startY, endX, endY) = box.astype("int")
            print(startX,startY)
        cv2.imshow("detections", frame)
        PROCESSINGFLAG = PROCESSINGFLAG + 1
    if PROCESSINGFLAG == 4:
        PROCESSINGFLAG = 0
    else:
        cv2.imshow("detections", frame)
        PROCESSINGFLAG = PROCESSINGFLAG + 1