#Import installed packages, cvzone and tello from djitellopy
import cv2
from djitellopy import tello
import cvzone

#This is for webcam testing
thres = 0.6                         #confidence threshold 60%
nmsThres =0.2                 #removes duplicates within bounding box for instances in which its detected more than once
cap = cv2.VideoCapture(0)
cap.set(3, 640)                     #This is for the video capture with prop ID for width 3 and value 640
cap.set(4, 400)                     #This is for the height

classNames = []                   #to import all the classes to the empty list to which the files will be added
classFile = 'coco.names'          #the file with the names
#import the file to the empty list
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')
print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'        #for the configuration file
weightsPath = "frozen_inference_graph.pb"                          #for the weight file

#load in a network
net = cv2.dnn_DetectionModel(weightsPath,configPath)          #to link to the files
net.setInputSize(320,320)                                   #ln 24-27 to set configuration parameters
net.setInputScale(1.0/127.5)                                #half of 8bits
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

 #The tello drone programming. ln 32 shows battery level, 33 turns stream on, 34 turns stream off, 36 take off and 37 climb
#me = tello.Tello()
#me.connect()
#print(me.get_battery())
#me.streamoff()
#me.streamon()

#me.takeoff()
#me.move_up(80)


while True:
    success, img = cap.read()
    #img = me.get_frame_read().frame              ~this will be used when using the drone instead of webcam
    classIds,confs,bbox = net.detect(img,confThreshold=thres, nmsThreshold=nmsThres)   #to return classId, confidence and bounding box found
    try:
        for classId, conf, box in zip(classIds.flatten(),confs.flatten(), bbox):           #unpack using zip method
            cvzone.cornerRect(img,box)                                                     #custom resctangle
            cv2.putText(img, f'{classNames[classId-1].upper()} {round(conf * 100,2)}',     #This displays text for the identified object class
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0,255,0), 2)

    except:
        pass
    #me.send_rc_controls(0, 0, 0, 0)         #for drone use

    cv2.imshow("Image",img)
    cv2.waitKey(1)