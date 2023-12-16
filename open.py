import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import serial,time

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

path = "imagesattendance"
images = []
classnames=[]
mylist=os.listdir(path)
print(mylist)
for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)

#ArduinoSerial=serial.Serial('com3',9600,timeout=0.1)
time.sleep(1)

def findEncodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('attend.csv','w') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodelistknown=findEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

    # imgS = cv2.resize(img,(0,0),None,0.25,0.25)
img=cv2.imread("TEST22.jpg")
imgS = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

facesCurFrame=face_recognition.face_locations(imgS)
encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
    matches = face_recognition.compare_faces(encodelistknown,encodeFace)
    faceDis= face_recognition.face_distance(encodelistknown,encodeFace)
    print(faceDis)
    matchIndex=np.argmin(faceDis)
    print(faceLoc)
    if  matches[matchIndex]:
        name=classnames[matchIndex].upper()
        print(name)
        y1,x2,y2,x1=faceLoc
        draw_border(img,(x1,y1),(x2,y2),(0,255,0),4,20,20)
        # cv2.rectangle(img, (x1,y2-20), (x2, y2), (0, 255, 0),cv2.FILLED)
        cv2.putText(img,name,(x1,y2+25),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,0,255),2)
        #markAttendance(name)
        ret, frame= cap.read()
        frame=cv2.flip(frame,1)  #mirror the image
        #print(frame.shape)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray,1.1,6)  #detect the face
        for x,y,w,h in faces:
        #sending coordinates to Arduino
            string='X{0:d}Y{1:d}'.format((x+w//2),(y+h//2))
            print(string)
            #ArduinoSerial.write(string.encode('utf-8'))
    # cv2.imshow('img',frame)

    else:
        y1,x2,y2,x1=faceLoc
        draw_border(img,(x1,y1),(x2,y2),(0,255,0),4,20,20)
        # cv2.rectangle(img, (x1,y2-20), (x2, y2), (0, 255, 0),cv2.FILLED)
        #cv2.putText(img,"No Data",(x1,y2+20),cv2.FONT_HERSHEY_PLAIN,1.1,(0,0,255),2)

    

cv2.imshow('Webcam',img)
cv2.waitKey(0)

