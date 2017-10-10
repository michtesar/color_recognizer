from __future__ import print_function

import os
import tensorflow as tf
import cv2
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
        training_set.append(cv2.imread(img_file, 1).reshape(1, 2764800))
        training_labels.append(label_name[0])
        n_files += 1

import tensorflow as tf

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

# Parameters
learning_rate = 0.001
training_epochs = n_files
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 2764800 # MNIST data input (img shape: 28*28)
n_classes = 8 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("int32")

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        batch_x = training_set[epoch]
        batch_y = y[epoch]
        _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
        # Compute average loss
        avg_cost += c / n_files
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))