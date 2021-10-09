#!/usr/bin/env python2

import sys

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def vidcad():
    global bridge
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False):  
        print "Error opening video  file"

    stream = rospy.Publisher("video",Image,queue_size=1)
    rospy.init_node('image_converter', anonymous=True)

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            stream.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
   
  
        else:  
            break
 
    cap.release()

if __name__ == '__main__':
    bridge = CvBridge()
    try:
        vidcad()
        
    except rospy.ROSInterruptException:
        print("Shutting down")
        
   
    



