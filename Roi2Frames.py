import cv2
print(cv2.__version__)
import numpy as np
from adafruit_servokit import ServoKit

#frame1 containes [pan,tilt,roi[x,y,w,h]] 
global frame1 
frame1 = np.zeros(2)
print(frame1)


#display width and height
dispW=640
dispH=480

#camera settings
camSet='nvarguscamerasrc !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv  ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)

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
        goFlag = 1


#define servo channels
kit=ServoKit(channels=16) 
pan=90
tilt=90
kit.servo[0].angle=pan
kit.servo[1].angle=tilt
 

def nothing(x):
    pass

def create_hsv_trackbars(): 
    cv2.namedWindow('Trackbars')
    cv2.moveWindow('Trackbars',1320,0)
    
    cv2.createTrackbar('hueLower', 'Trackbars',46,179,nothing)
    cv2.createTrackbar('hueUpper', 'Trackbars',86,179,nothing)
    
    cv2.createTrackbar('hue2Lower', 'Trackbars',50,179,nothing)
    cv2.createTrackbar('hue2Upper', 'Trackbars',0,179,nothing)
    
    cv2.createTrackbar('satLow', 'Trackbars',157,255,nothing)
    cv2.createTrackbar('satHigh', 'Trackbars',255,255,nothing)
    cv2.createTrackbar('valLow','Trackbars',100,255,nothing)
    cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)

def create_panTilt_trackbars():
    cv2.createTrackbar('Pan', 'Trackbars',90,180,nothing)
    cv2.createTrackbar('Tilt', 'Trackbars',90,180,nothing)




#create the taskbars
create_hsv_trackbars()
create_panTilt_trackbars()

width=cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('width:',width,'height:',height)



print(frame1)
while True:   
    ret, frame = cam.read()
    #frame=cv2.imread('smarties.png')
 
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hueLow=cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp=cv2.getTrackbarPos('hueUpper', 'Trackbars')

    hue2Low=cv2.getTrackbarPos('hue2Lower', 'Trackbars')
    hue2Up=cv2.getTrackbarPos('hue2Upper', 'Trackbars')

    Ls=cv2.getTrackbarPos('satLow', 'Trackbars')
    Us=cv2.getTrackbarPos('satHigh', 'Trackbars')

    Lv=cv2.getTrackbarPos('valLow', 'Trackbars')
    Uv=cv2.getTrackbarPos('valHigh', 'Trackbars')
    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])
 
    l_b2=np.array([hue2Low,Ls,Lv])
    u_b2=np.array([hue2Up,Us,Uv])
 
    FGmask=cv2.inRange(hsv,l_b,u_b)
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)
    FGmaskComp=cv2.add(FGmask,FGmask2)


#Create black white mask for hsv     
    cv2.imshow('FGmaskComp',FGmaskComp)
    cv2.moveWindow('FGmaskComp',0,530)

    cv2.getTrackbarPos('Pan','Trackbars')
    cv2.getTrackbarPos('Tilt', 'Trackbars')

#initilize frame 1 
    if cv2.waitKey(1)==ord('1'):
        frame1[0]=cv2.getTrackbarPos('Pan','Trackbars')
        frame1[1]=cv2.getTrackbarPos('Tilt', 'Trackbars') 
        pan = frame1[0]
        tilt = frame1[1] 
        kit.servo[0].angle=pan
        kit.servo[1].angle=tilt 
        print(frame1)


    contours,_=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=50:

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            objX=x+w    # X position of the right bb of the object we tracking
            objY=y+h    # Y position of the top of the object we tracking
            
            #error between the center of the frame to the object
            errorPan_right = objX-width/2
            errorPan_left = x - width/2   
            errorTilt=objY-height/2 
            
            
            if abs(errorPan_right)>width/2-5:
                pan=pan-errorPan_right*2/32
                print("errorpan right : ", errorPan_right)
                print("pan : " ,pan)
            if abs(errorPan_left)>width/2-5:
                pan=pan-errorPan_left*2/32
                print("errorpan left : ", errorPan_left)
                print("pan : " ,pan)

            if abs(errorTilt)>height/2-10:
                tilt=tilt-errorTilt/75
 
 
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
            break        
 

    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,0)
    
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()