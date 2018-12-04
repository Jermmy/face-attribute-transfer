import pickle

with open('data/aus_openface.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin')
    print(data)

import face_recognition
import cv2

image = face_recognition.load_image_file("S059_002_00000006.png")
l = face_recognition.face_locations(image, model='cnn')[0]

print(l)
height = l[2] - l[0]
cv2.imshow('image', image)
face = image[max(l[0]-height//4, 0):l[2], l[3]:l[1]]
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
cv2.imshow('face', face)
cv2.waitKey()
