# online-Face-Detection-system
import cv2
from random import randrange as r
trainedData=cv2.CascadeClassifier('haar.xml ')

webcam=cv2.VideoCapture(0)

while True:
    success,frame=webcam.read()
    
    grayimg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faceCoordinates=trainedData.detectMultiScale(grayimg)

    for x,y,w,h in faceCoordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(r(0,256),r(0,256),r(0,256)),2)
    
    cv2.imshow('window',frame)
    key=cv2.waitKey(1)
    
    if(key==81 or key==113):
        break
    
webcam.release()

