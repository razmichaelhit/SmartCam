import numpy as np
import argparse
import time
import cv2
import imutils


prototext = "~/Desktop/test_face/mtcnn_facenet_cpp_tensorRT/mtCNNModels/det1_relu.prototxt"
model = "~/Desktop/test_face/mtcnn_facenet_cpp_tensorRT/mtCNNModels/det1_relu.caffemodel"

# prototext = "deploy_lowers.prototxt"
# model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototext, model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# initialize the video stream and allow the camera sensor to warm up

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(DISPLAY_WIDTH)+', height='+str(DISPLAY_HEIGHT)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)

time.sleep(2.0)
# loop over the frames from the video stream
while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    net.setInput(blob) 
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.3:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
cam.release()