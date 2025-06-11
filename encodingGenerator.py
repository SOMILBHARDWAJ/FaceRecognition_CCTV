#we have to code and generate data for 128 measurement scales and store it in a file , then import it into our facerecognition then it will detect and display
import cv2
import face_recognition
import os

#import faces then encode it and save it into a list and dump it using pickle lib
import pickle

# Importing the faces into a list
folderPath = 'Images'
facePathList = os.listdir(folderPath)
imgList = []
faceIDs = []

for path in facePathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #get the id from faces removing .png
    # print(os.path.splitext(path)[0])
    faceIDs.append(os.path.splitext(path)[0])

print(faceIDs)


#fun that would consume these faceIDs and give respective encodings
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
         #change color BGR--RGB for face-recog
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         #find encodings
         encodings = face_recognition.face_encodings(img)[0]
         encodeList.append(encodings)

    return encodeList

print("Encoding started..")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,faceIDs]
print("Encoding complete")

#put all this is pickle file then use it while face recognition
file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")



