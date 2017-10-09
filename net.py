import os
import tensorflow as tf
import cv2

data_dir = 'training_set/'
class_names = [
    "Black",
    "White",
    "Red",
    "Green",
    "Blue",
    "Orange",
    "Yellow",
    "Purple",
]

if (os.path.isdir(data_dir)):
    print('Training set folder was found')
else:
    print('There is no traing set folder in root')
training_set = list()
for file in os.listdir(data_dir):
    if file.endswith(".jpg"):
        img_file = os.path.join(data_dir, file)
        training_set.append(cv2.imread(img_file, 0))
