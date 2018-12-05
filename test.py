import pickle
import face_recognition
import cv2
import numpy as np

# image = face_recognition.load_image_file("S059_002_00000006.png")
# l = face_recognition.face_locations(image, model='cnn')[0]
#
# print(l)
# height = l[2] - l[0]
# cv2.imshow('image', image)
# face = image[max(l[0]-height//4, 0):l[2], l[3]:l[1]]
# face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# cv2.imshow('face', face)
# cv2.waitKey()


image = cv2.imread('S010_001_00000008.png')
# landmarks = np.loadtxt('S010_001_00000008.csv', delimiter=',', skiprows=1)
# landmarks = landmarks[2:].astype(np.int32)

# for i in range(68):
#     cv2.circle(image, (landmarks[i], landmarks[i+68]), radius=2, color=(255, 0, 0), thickness=2)
# cv2.imwrite('image.png', image)


with open('S010_001_00000008_landmarks.txt', 'r') as f:
    landmarks = []
    for line in f.readlines():
        line = line.strip().split('   ')
        landmarks += [(int(float(line[0])), int(float(line[1])))]

for l in landmarks:
    cv2.circle(image, l, radius=2, color=(255, 0, 0), thickness=2)
landmarks = np.array(landmarks)
print(landmarks)
tl = np.min(landmarks, axis=0)
br = np.max(landmarks, axis=0)
print(tl)
cv2.circle(image, (tl[0], tl[1]), radius=2, color=(0, 255, 0), thickness=3)
cv2.circle(image, (br[0], br[1]), radius=2, color=(0, 255, 0), thickness=3)
cv2.imwrite('image1.png', image)