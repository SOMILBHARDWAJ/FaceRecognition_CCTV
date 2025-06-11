import os
import pickle

import cvzone
import numpy as np
import cv2
import face_recognition

from encodingGenerator import encodeListKnownWithIds

# webcam setup
cap = cv2.VideoCapture(1)
cap.set(3, 640) #width
cap.set(4, 480) #height

#background img
imgBackground = cv2.imread('Resources/background.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#print(len(imgModeList))

#Load the encoding file
print("Loading Encode file")
file = open('EncodeFile.p','rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown , faceIDs = encodeListKnownWithIds
#print(faceIDs)
print("Encoded Files Loaded..")


while True:
    success, img = cap.read()

    #small image for better computation
    imgS = cv2.resize(img, (0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)


    #current small face and its encoding
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS,faceCurrentFrame)

    #overlay camera on background img 162 , 55 are starting points (H ,W)
    imgBackground[162:162+480,55:55+640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]


    #we will loop and compare our current face encoding to the encodingGenerator's encoding

    #encodeCUrrentFrame--> encFrame and faceIDs-->faceLoc
    for encodeFace , faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        #lower the faceDistance better it is
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("Matches:",matches)
        # print("FaceDistance:",faceDistance)

        matchIndex = np.argmin(faceDistance)
        #matchIndex{face_index5}==matches{true}
        if matches[matchIndex]:
            print("known face detected")
            print("Matched:", faceIDs[matchIndex])
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)


    cv2.imshow('Face Attendance', imgBackground)
    cv2.waitKey(1)