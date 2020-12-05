import cv2 
import face_recognition
import numpy as np 
import os
import glob
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


video_capture = cv2.VideoCapture(0) 
  
known_face_encodings = []
known_face_names = []
dirname = os.path.dirname(__file__)
path = os.path.join(dirname,'known_people\\')

list_of_files = [f for f in glob.glob(path+'*.jpg')]
number_files = len(list_of_files)

names = list_of_files.copy()
patter = '[0-9]'

for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    known_face_encodings.append(globals()['image_encoding_{}'.format(i)])
    names[i] = names[i].replace("known_people\\","")
    names[i] = re.sub(patter,'',names[i])
    known_face_names.append(names[i])


le = LabelEncoder()
known_face_names = le.fit_transform(known_face_names)


X_train, X_test, y_train, y_test = train_test_split(known_face_encodings,known_face_names,test_size=0.1, random_state=0)
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,y_train)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rbg_small_frame = small_frame[:,:,::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rbg_small_frame)
        face_encodings = face_recognition.face_encodings(rbg_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            temp = classifier.predict([face_encoding])
            temp = le.inverse_transform(temp)
            name = temp[0]
            face_names.append(name)

    process_this_frame = not process_this_frame

    for(top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *=4
        bottom *= 4
        left *=4

        cv2.rectangle(frame, (left,top), (right, bottom), (0,0,255), 2)
        cv2.rectangle(frame, (left,bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255),1)

    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()