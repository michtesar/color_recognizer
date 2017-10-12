import os
import cv2
#from sklearn.linear_model import perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np

# shape of image is 36*64*3, which is 6912 respahed into vector

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

def learn():
    print('Loading previous dataset to learn')
    n_files = 0
    training_set = list()
    training_labels = list()
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            img_file = os.path.join(data_dir, file)
            label_name = str(file).split('_')
            training_set.append(cv2.imread(img_file, 1).reshape(6912))
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

    #net = perceptron.Perceptron(max_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
    net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 8), random_state=1, verbose=True)

    print('\nLearning...\n')
    net.fit(x, y)

    print('MLP has already learned previous instances')

    return net

def identify_color(src_image, net):
    image_resised = src_image.reshape(1, 6912)
    p = net.predict(image_resised)

    return str(class_names[int(p)])

def weights_to_image(net):
    # Only first layer is extracted
    weights = net.coefs_[0]
    weights_average = weights.mean(axis=1)
    weights_image = weights_average.reshape(64, 36, 3)
    np.save('weights.npy', weights_image)
    return weights_image