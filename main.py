import cv2
import face_recognition
import os
import numpy as np

path="train"
images = []
classNames = []
myList =os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def Mahoa (images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)

imgTest = cv2.imread("test/test.png")

while True:
    framS = cv2.resize(imgTest,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    facecurFrame = face_recognition.face_locations(framS)
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.5 :
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown"


        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2 , x2 * 2 , y2 * 2 , x1 * 2
        cv2.rectangle(imgTest,(x1,y1), (x2,y2),(0,255,0),3)
        cv2.putText(imgTest,name,(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

    cv2.imshow('imgtest',imgTest)
    if cv2.waitKey(1) == ord("q"):
        break
imgTest.release()
cv2.destroyAllWindows()

