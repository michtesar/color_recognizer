import os
import tensorflow as tf

data_dir = 'training_set/'
if (os.path.isdir(data_dir)):
    print('Training set folder was found')
else:
    print('There is no traing set folder in root')

