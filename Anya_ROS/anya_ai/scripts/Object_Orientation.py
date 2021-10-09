#!/usr/bin/env python
import cv2
import numpy as np
import glob
import random
import sys
import os
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#--- Define our Class
def callback(data):
    global catch
    global bridge
    # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
    catch = bridge.imgmsg_to_cv2(data, "bgr8")

def orient_yolo():

    rospy.Subscriber("camera/rgb/image_raw",Image, callback)
    rospy.init_node('orient_yolo', anonymous=True)

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

                    mask = catch[y-20:y+h+20,x-20:x+w+20]

                    print("img for rectangle {}".format(img.shape))
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 1, color, 1)

                    cv2.imshow("Masked", mask)
                    # convert the image to grayscale
                    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("Gray", gray)
                    edged = cv2.Canny(gray, 30, 150)
                    cv2.imshow("Edged", edged)
                    X=[]
                    Y=[]
                    for i in range(0,mask.shape[0]):
                        for j in range(0,mask.shape[1]):
                            if(edged[i][j]==255):
                                X.append(j)
                                Y.append(-i)
                    Xlen=len(X)
                    Ylen=len(Y)

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

                    # plt.plot(X, y_predicted, color='r')
                    # plt.show()



            cv2.imshow("Image",  cv2.resize(img,(640,480)))

        except:
            kill+=1
            print("Yolo exception")
            if(kill==500000):
                print("Killing amma")
                quit()
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
        orient_yolo()
        
        
    except rospy.ROSInterruptException:
        print("Shutting down")