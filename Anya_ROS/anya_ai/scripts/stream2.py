#!/usr/bin/env python2
import sys

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    global catch
    global bridge
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catch = bridge.imgmsg_to_cv2(data, "bgr8")
#--- Define our Class
def show():

    rospy.Subscriber("video",Image, callback)
    rospy.init_node('image_s', anonymous=True)
    usr = input("Enter a value")
    while(True):
        try: 
            cv2.imshow('Frame', catch)
        except:
            pass
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

#--------------- MAIN LOOP

def main():
    rospy.Subscriber("video",Image, callback)
    rospy.init_node('image_s', anonymous=True)
    usr = input("Enter a value")

if __name__ == '__main__':
    catch = None
    bridge = CvBridge()
    try:
        main()
        
        
    except rospy.ROSInterruptException:
        print("Shutting down")
        
    #--- In the end remember to close all cv windows
    

