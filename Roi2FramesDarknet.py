import cv2
import imutils
import time
import numpy as np
from adafruit_servokit import ServoKit

# display width and height
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FOV = 40  # Servo FOV to change frame
# camera settings
camSet = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(
    DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam = cv2.VideoCapture(camSet)

# Parameters to mouse click event
goFlag = 0
x1 = 0
x2 = 0
y1 = 0
y2 = 0

# define servo channels
kit = ServoKit(channels=16)

# draw ROI - Region Of Interest


def mouse_click(event, x, y, flags, params):
    global x1, y1, x2, y2
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

# Create Pan Tilt trackbars for first initialization


def create_panTilt_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.moveWindow('Trackbars', 1320, 0)
    cv2.createTrackbar('Pan', 'Trackbars', 90, 180, nothing)
    cv2.createTrackbar('Tilt', 'Trackbars', 90, 180, nothing)

# Class camera for change position


class imxCamera:
    def __init__(self, pan, tilt):
        self.pan = pan
        self.tilt = tilt

    def changePosition(pan, tilt):
        if pan > 180 or tilt > 180:
            print("pan or tilt cannot be more than 180")
            return
        kit.servo[0].angle = pan
        kit.servo[1].angle = tilt


# Class camera for each frame of the session
class FrameView:
    def __init__(self):
        self.frame_pan = 90
        self.frame_tilt = 90
        self.roi = np.zeros(4)

    def __str__(self):
        return'Roi: {self.roi} pan {self.frame_pan} tilt: {self.frame_tilt}'.format(self=self)

# Set the roi from the user
    def setRoi(self):
        ret, frame = cam.read()
        cv2.imshow('nanoCam', frame)
        cv2.moveWindow('nanoCam', 0, 0)
        cv2.setMouseCallback('nanoCam', mouse_click)
        if goFlag == 1:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            region_of_interest = cv2.rectangle(
                frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.imshow('nanoCam', region_of_interest)

# get the roi to the frame roi
    def getRoi(self):
        self.roi[0] = x1
        self.roi[1] = x2
        self.roi[2] = y1
        self.roi[3] = y2


# show the roi

    def showRoi(self, frame):
        frame = cv2.rectangle(frame, (int(self.roi[0]), int(
            self.roi[2])), (int(self.roi[1]), int(self.roi[3])), (255, 0, 0), 3)
        region_of_interest = cv2.rectangle(frame, (int(self.roi[0]), int(
            self.roi[2])), (int(self.roi[1]), int(self.roi[3])), (255, 0, 0), 3)
        cv2.imshow('nanoCam', region_of_interest)

# init the first frame
    def initFrame1(self):
        while True:
            self.frame_pan = cv2.getTrackbarPos('Pan', 'Trackbars')
            self.frame_tilt = cv2.getTrackbarPos('Tilt', 'Trackbars')
            imxCamera.changePosition(self.frame_pan, self.frame_tilt)
            FrameView.setRoi(self)
            if cv2.waitKey(1) == ord('1'):
                FrameView.getRoi(self)
                break

# init any other frame with frame_fame_idx
    def initRightFrame(self, frame_fame_idx, pan, tilt):
        self.frame_tilt = tilt
        self.frame_pan = pan
        self.frame_pan = self.frame_pan - FOV*(int(frame_fame_idx)-1)
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        while True:
            FrameView.setRoi(self)
            if cv2.waitKey(1) == ord(frame_fame_idx):
                FrameView.getRoi(self)
                break


# show the frame includes the roi


    def showFrame(self, frame):
        imxCamera.changePosition(self.frame_pan, self.frame_tilt)
        FrameView.showRoi(self, frame)
        # is_suspect()


# face detection Deep Nueral Network for first tracking object
def faceDetector(frame_right):
    LABELS = open('Person.names').read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    confidence_percent = 0.5
    threshold = 0.3
    # FRAME_CHANGED_FLAG - if frame has been changed , stop tracking the object for 10 loops, keep from latency bugs
    FRAME_CHANGED_FLAG = 0
    PROCESSING_FLAG = 0  # to reduce latency we processing 2 of every 4 frames
    fame_idx = 0
    # Loading DNN Network to object detection
    weights = "yolov4.cfg"
    cfg = "yolov4.weights"
    net = cv2.dnn.readNetFromDarknet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    while True:
        PROCESSING_FLAG = PROCESSING_FLAG + 1
        ret, frame = cam.read()
        # change shape for dnn
        (h, w) = frame.shape[:2]
        # Showing the frame by the frame indx
        frame_right[fame_idx].showFrame(frame)
        if PROCESSING_FLAG == 2 or PROCESSING_FLAG == 3:
            # determine only the *output* layer names that we need from YOLO
#           blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),	swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward()
            end = time.time()
            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            # show timing information on YOLO
           # print("[INFO] YOLO took {:.6f} seconds".format(end - start)
            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > confidence_percent:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        idxs = cv2.dnn.NMSBoxes(
                            boxes, confidences, confidence_percent, threshold)
                        if len(idxs) > 0:
                            # loop over the indexes we are keeping
                            for i in idxs.flatten():
                                # extract the bounding box coordinates
                                (x, y) = (boxes[i][0], boxes[i][1])
                                (w, h) = (boxes[i][2], boxes[i][3])
                                # draw a bounding box rectangle and label on the image
                                color = [int(c) for c in COLORS[classIDs[i]]]
                                cv2.rectangle(
                                    image, (x, y), (x + w, y + h), color, 2)
                                text = "{}: {:.4f}".format(
                                    LABELS[classIDs[i]], confidences[i])
                                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, color, 2)
                        # show the output image
                        cv2.imshow("Image", image)

                # Tracking after the subject with the camera
                if FRAME_CHANGED_FLAG > 0:
                    FRAME_CHANGED_FLAG = FRAME_CHANGED_FLAG+1
                    if FRAME_CHANGED_FLAG > 10:
                        FRAME_CHANGED_FLAG = 0
                    continue
                if endX > DISPLAY_WIDTH-2:
                    fame_idx = fame_idx + 1
                    FRAME_CHANGED_FLAG = 1
                    print(fame_idx)
                    continue
                if fame_idx == 0:
                    continue
                if startX < 2:
                    fame_idx = fame_idx - 1
                    FRAME_CHANGED_FLAG = 1
                    continue

            cv2.imshow("nanoCam", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        if PROCESSING_FLAG == 4:
            PROCESSING_FLAG = 0
        else:
            cv2.imshow("nanoCam", frame)
            # if the `q` key was pressed, break from the loop
            if cv2.waitKey(1) == ord("q"):
                break
    cam.release()
    cv2.destroyAllWindows()


# main func
def main():
    create_panTilt_trackbars()

    # transverse array of frames
    frame_right = []

    frame_right.append(FrameView())
    frame_right[0].initFrame1()
    print(frame_right[0])

    frame_right.append(FrameView())
    frame_right[1].initRightFrame(
        '2', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[1])

    frame_right.append(FrameView())
    frame_right[2].initRightFrame(
        '3', frame_right[0].frame_pan, frame_right[0].frame_tilt)
    print(frame_right[2])

    # detct Faces
    faceDetector(frame_right)
    return


main()
