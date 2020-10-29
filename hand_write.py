import os

import cv2
import numpy

from train_model import train_return_model, predict_image

cap = cv2.VideoCapture(0)

points = []

flag = 0

retval = os.getcwd()
print(retval)
model = train_return_model()
first_time = 1
while(1):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    finalImg = frame
    key = cv2.waitKey(1)
    filterFrame = cv2.GaussianBlur(frame, (35, 35), 25)
    #
    hsvFrame = cv2.cvtColor(filterFrame, cv2.COLOR_BGR2HSV)
    finalImg = hsvFrame
    #
    lower_bound = numpy.array([30, 100, 100])
    upper_bound = numpy.array([80, 255, 255])
    #
    threshImg = cv2.inRange(hsvFrame, lower_bound, upper_bound)
    finalImg = threshImg
    #
    contours, heirarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    finalImg = cv2.bitwise_and(frame, frame, mask=threshImg)
    #
    finalImg = cv2.drawContours(finalImg, contours, -1, (255, 0, 0), 1)
    #
    c = 0

    X = 0
    Y = 0


    if flag == 1:
        first_time = 0
        for item in contours:
            for i in item:
                X += i[0][0]
                Y += i[0][1]
                c += 1

        try:
            points.append([int(X / c), int(Y / c)])
        except:
            pass

    if (key & 0xFF == ord('s')) and flag == 0:
        flag = 1

    elif key & 0xFF == ord('s') and flag == 1:
        flag = 0

    elif key & 0xFF == ord('c'):
        points = []

    for p in points:
        cv2.circle(finalImg, tuple(p), 20, (0, 0, 255), -1)

    if(flag==0 and first_time==0):
        cv2.imwrite('temp.png', finalImg)
        number = predict_image(finalImg, model)
        print(number)
        finalImg = cv2.putText(finalImg, str(number) , (100,200), cv2.FONT_HERSHEY_SIMPLEX,
                            4, (255, 0, 0), 4, cv2.LINE_AA)



    cv2.imshow('HSV', finalImg)

    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
