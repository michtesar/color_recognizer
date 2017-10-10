import os
import cv2
from sklearn.linear_model import perceptron
import numpy as np

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

n_files = 0
training_set = list()
training_labels = list()
for file in os.listdir(data_dir):
    if file.endswith(".jpg"):
        img_file = os.path.join(data_dir, file)
        label_name = str(file).split('_')
        training_set.append(cv2.imread(img_file, 1).reshape(2764800))
        training_labels.append(label_name[0])
        n_files += 1


def integerize(data):
    Y = list()
    for i in range(n_files):
        a = data[i]
        if a == 'Black':
            Y.append(0)
        elif a == 'White':
            Y.append(1)
        elif a == 'Red':
            Y.append(2)
        elif a == 'Green':
            Y.append(3)
        elif a == 'Blue':
            Y.append(4)
        elif a == 'Orange':
            Y.append(5)
        elif a == 'Yellow':
            Y.append(6)
        elif a == 'Purple':
            Y.append(7)    
    return Y

y = integerize(training_labels)
x = training_set

print(np.shape(x))

net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
net.fit(x, y)

predict = training_set.append(cv2.imread('training_set/Blue_training.jpg', 1).reshape(2764800))

p = net.predict(predict)
print(p)