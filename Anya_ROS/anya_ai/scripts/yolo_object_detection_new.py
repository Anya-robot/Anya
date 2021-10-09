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


def callback(data):
    global catch
    global bridge
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catch = bridge.imgmsg_to_cv2(data, "bgr8")

def yolo():

    rospy.Subscriber("camera/rgb/image_raw",Image, callback)
    rospy.init_node('yolo', anonymous=True)

    net = cv2.dnn.readNet("/home/bharathchandra/catkin_ws/src/anya_ai/scripts/Resources/yolov3_custom_7000.weights", 
    "/home/bharathchandra/catkin_ws/src/anya_ai/scripts/Resources/yolov3_custom_7000.cfg")

    classes = ["Pen", "Apple", "Banana", "Orange", "Medicine_Bottle"]

    # Images path




    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    kill=0
    while(True):
        try: 
            # cv2.imshow('Frame', catch)
            img = cv2.resize(catch, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
                    if confidence > 0.2:
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
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 2)


            cv2.imshow("Image",  cv2.resize(img,(640,480)))

        except Exception as e: 
            print(e)
            kill+=1
            #print("Yolo exception")
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
    try:
        yolo()
        
        
    except rospy.ROSInterruptException:
        print("Shutting down")