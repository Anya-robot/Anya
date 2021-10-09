#!/usr/bin/env python

import sys
import numpy as np
import time

#hi ra madhav
#hi ra leader

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(data):
    global catch
    global bridge
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catch = bridge.imgmsg_to_cv2(data,"bgr8")
#--- Define our Class
def show():

    rospy.Subscriber("camera/rgb/image_raw",Image, callback)
    rospy.init_node('image_show', anonymous=True)

    while(True):
        
        mask = np.zeros(catch.shape,np.unit8)
        mask[y-20:y+h+20,x-20:x+w+20] = catch[y-20:y+h+20,x-20:x+w+20]
        cv2.imshow("Masked", mask)
        # convert the image to grayscale
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", gray)
        edged = cv2.Canny(gray, 30, 150)
        X=[]
        Y=[]
        for i in range(0,480):
            for j in range(0,640):
                if(edged[i][j]==255):
                    X.append(j)
                    Y.append(-i)
        Xlen=len(X)
        Ylen=len(Y)
        cv2.imshow("Edged", edged)

        # generate random data-set
        X = np.array(X)
        X = X.reshape(Xlen,1)
        Y = np.array(Y)
        Y = Y.reshape(Ylen,1)
        print(X.shape,Y.shape)
        # sckit-learn implementation

        # Model initialization
        regression_model = LinearRegression()
        # Fit the data(train the model)
        regression_model.fit(X, Y)
        # Predict
        y_predicted = regression_model.predict(X)

        # model evaluation
        rmse = mean_squared_error(Y, y_predicted)
        r2 = r2_score(Y, y_predicted)

        # printing values
        print('Slope:' ,regression_model.coef_)
        print('Intercept:', regression_model.intercept_)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)

        # plotting values

        # data points
        plt.scatter(X, Y, s=3)
        plt.xlabel('x')
        plt.ylabel('y')

        # predicted values


        index1 = np.where(X == min(X))
        index2 = np.where(X == max(X))
        cv2.line(gray, (X[index1[0][0]],-1*y_predicted[index1[0][0]]), (X[index2[0][0]],-1*y_predicted[index2[0][0]]), 255, 6)
        cv2.imshow("Orientations", gray)

        plt.plot(X, y_predicted, color='r')
        plt.show()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

#--------------- MAIN LOOP


if __name__ == '__main__':
    catch = None
    bridge = CvBridge()
    try:
        show()
        
        
    except rospy.ROSInterruptException:
        print("Shutting down")
        
    #--- In the end remember to close all cv windows
    

       