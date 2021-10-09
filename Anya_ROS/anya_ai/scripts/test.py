#!/usr/bin/env python2
import sys

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def test():
    cap = cv2.VideoCapture('/home/pranav/catkin_ws/src/opencv/scripts/auto.mp4')
    if (cap.isOpened()== False):  
        print("Error opening video  file") 

    # stream = rospy.Publisher("video",Image,queue_size=1)
    rospy.init_node('video_test', anonymous=True)

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            cv2.imshow('Frame', frame) 
   
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
   
  
        else:  
            break
    cap.release()
    cv2.destroyAllWindows()



 

 
if __name__ == '__main__':
    try:
        test()
        
    except rospy.ROSInterruptException:
        print("Shutting down")

    