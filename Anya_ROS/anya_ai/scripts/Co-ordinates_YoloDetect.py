#!/usr/bin/env python
import cv2
import numpy as np
import glob
import random
import sys
import os
import time

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callbackd(data):
    global catchd
    global bridged
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catchd = bridge.imgmsg_to_cv2(data,"32FC1")
#--- Define our Class
def callback(data):
    global catch
    global bridge
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catch = bridge.imgmsg_to_cv2(data, "bgr8")

def depth_yolo():

    rospy.Subscriber("camera/rgb/image_raw",Image, callback)
    rospy.Subscriber("camera/depth/image_raw",Image, callbackd)
    rospy.init_node('depth_yolo', anonymous=True)

    net = cv2.dnn.readNet("/home/bharathchandra/catkin_ws/src/anya_ai/scripts/Resources/yolov3_custom_final.weights", 
    "/home/bharathchandra/catkin_ws/src/anya_ai/scripts/Resources/yolov3_custom.cfg")
    print("Yolo model loaded")

    classes = ["Pen", "Apple", "Banana", "Orange"]

    # Images path


    kill=0

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    while(True):
        try: 
            # cv2.imshow('Frame', catch)
            #print("Catch {}".format(catch.shape))
            img=catch
            #img = cv2.resize(catch, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            #print("Img {}".format(img.shape))
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            #print("Blob {}".format(blob.shape))
 
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Showing informations on the screen
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        print(class_id)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])+str(confidences[i])
                    color = colors[class_ids[i]]
                    print("img for rectangle {}".format(img.shape))
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 1, color, 1)
                    #print("Catchd {}".format(catchd.shape)
                    # x1=256*x//640
                    # y1=192*y//480
                    # x2=256*(x+w)//640
                    # y2=192*(y+h)//480
                    # W=x2-x1
                    # H=y2-y1
                    crop_img = catchd[y+h//4:y+h-h//4,x+w//4:x+w-w//4]
                    na_crop_img=np.array(crop_img)
                    #avg_val=np.average(crop_imp)
                    #print(avg_val)
                    na_crop_img=np.nanmean(na_crop_img,axis=0)

                    xc=x+w//2
                    yc=y+h//2
                    zr=np.average(na_crop_img)
                    fx=551.7433199726613
                    cx=316.9042901766989
                    fy=521.9473349352442
                    cy=249.7252492404512

                    xr=((xc-cx)*zr)/fx
                    yr=-1*((yc-cy)*zr)/fy


                    print("(X,Y,Z) = ({},{},{}) ".format(xr,yr,zr))
                    #cv2.imshow("cropped", crop_img)
                    cv2.rectangle(catchd,(x+w//4, y+h//4), (x+ w -w//4, y+h-h//4),0,2)
                    # cv2.rectangle(catchd,(x1, y1), (x2, y2),0,2)
                    cv2.rectangle(catchd, (x, y), (x + w, y + h), 0, 2)
                    norm_catchd=catchd.copy()
                    cv2.normalize(catchd, norm_catchd, 0, 1, cv2.NORM_MINMAX)
                    cv2.imshow('Frame', norm_catchd)


            cv2.imshow("Image",  cv2.resize(img,(640,480)))

        except:
            kill+=1
            print("Exception")
            if(kill==500000):
                print("Killing amma")
                quit()
            pass
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Load Yolo
# Name custom object
# Insert here the path of your images
# loop through all the images


if __name__ == '__main__':
    catch = None
    bridge = CvBridge()
    catchd = None
    bridged = CvBridge()
    try:
        depth_yolo()
        
        
    except rospy.ROSInterruptException:
        print("Shutting down")