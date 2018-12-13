import pickle
import face_recognition
import cv2
import numpy as np
import copy

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


# image = cv2.imread('S005_001_00000001.png')
#
# with open('S005_001_00000001_landmarks.txt', 'r') as f:
#     landmarks = []
#     for line in f.readlines():
#         line = line.strip().split('   ')
#         landmarks += [(int(float(line[0])), int(float(line[1])))]
#
# image_landmark = copy.deepcopy(image)
# for l in landmarks:
#     cv2.circle(image_landmark, l, radius=2, color=(255, 0, 0), thickness=2)
# landmarks = np.array(landmarks)
#
# tl = np.min(landmarks, axis=0)
# br = np.max(landmarks, axis=0)
#
# cv2.circle(image_landmark, (tl[0], tl[1]), radius=2, color=(0, 255, 0), thickness=3)
# cv2.circle(image_landmark, (br[0], br[1]), radius=2, color=(0, 255, 0), thickness=3)
# cv2.imwrite('image_landmark.png', image_landmark)
#
# height = br[1] - tl[1]
# width = br[0] - tl[0]
#
# shift_up = max(tl[1] - height // 4, 0)
# shift_left = max(tl[0] - width // 10, 0)
#
# image_clip = copy.deepcopy(image[max(tl[1] - height // 4, 0): br[1],
#                 max(tl[0] - width // 10, 0): min(br[0] + width // 10, image.shape[1])])
# cv2.imwrite('image_clip.png', image_clip)
#
# landmarks[:,1] = landmarks[:,1] - shift_up
# landmarks[:,0] = landmarks[:,0] - shift_left
# for l in landmarks:
#     cv2.circle(image_clip, (l[0], l[1]), radius=2, color=(255, 0, 0), thickness=2)
# cv2.imwrite('image_clip_landmark.png', image_clip)
#
#
#
# scale_height = image_clip.shape[0] / 160
# scale_width = image_clip.shape[1] / 160
# image_scale = image[max(tl[1] - height // 4, 0): br[1],
#                 max(tl[0] - width // 10, 0): min(br[0] + width // 10, image.shape[1])]
# image_scale = cv2.resize(image_scale, (160, 160), interpolation=cv2.INTER_AREA)
# for l in landmarks:
#     cv2.circle(image_scale, (int(l[0] / scale_width), int(l[1] / scale_height)), radius=1, color=(255, 0, 0), thickness=1)
# cv2.imwrite('image_clip_scale.png', image_scale)


