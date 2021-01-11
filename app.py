import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
import os
os.environ["TF_XLA_FLAGS"]="--tf_xla_enable_xla_devices"


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# read images
img = cv2.imread("people.jpg")

img_h,img_w,img_channels = img.shape


# convert to gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# initialize detector
detector = MTCNN()

# detect faces
faces = detector.detect_faces(img)

print(faces[0]['keypoints'])


for i in faces:
    box = i['box']

    w = box[0]
    h = box[1]
    x = box[0]+box[2]
    y = box[1]+box[3]

    img = cv2.rectangle(img,(w,h),(x,y),(255,0,0),2)

    # Left Eye
    radius = 2
    color = (0, 0, 255)
    img = cv2.circle(img, i['keypoints']['left_eye'], radius, color)

    # Right Eye
    radius = 2
    color = (0, 0, 255)
    img = cv2.circle(img, i['keypoints']['right_eye'], radius, color)

    # Nose
    radius = 2
    color = (0, 0, 255)
    img = cv2.circle(img, i['keypoints']['nose'], radius, color)

    # Left Mouth
    radius = 2
    color = (0, 0, 255)
    img = cv2.circle(img, i['keypoints']['mouth_left'], radius, color)

    # Right Mouth
    radius = 2
    color = (0, 0, 255)
    img = cv2.circle(img, i['keypoints']['mouth_right'], radius, color)


cv2.imshow("face detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()