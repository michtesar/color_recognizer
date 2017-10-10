import numpy as np
from matplotlib import pyplot as plt

weights_image = np.load('weights.npy')

mean_R = weights_image[:,:,0].mean()
mean_G = weights_image[:,:,1].mean()
mean_B = weights_image[:,:,2].mean()

print(mean_R)
print(mean_G)
print(mean_B)