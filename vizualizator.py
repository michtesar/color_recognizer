import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2

def load_dummy_data():
    weights_image = np.load('weights.npy')
    return weights_image

def extract_rgb(data):
    mean_R = data[:,:,0].mean()
    mean_G = data[:,:,1].mean()
    mean_B = data[:,:,2].mean()

    return mean_R, mean_G, mean_B

def extract_rbg_vector(data):
    vector_R = data[:,:,0]
    vector_G = data[:,:,0]
    vector_B = data[:,:,0]

    return vector_R, vector_G, vector_B

def extract_average_color(data):
    average_color = data.mean(axis=2)

    return average_color

def normalize_to_greyscale(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    normalized = scaler.transform(data)

    return normalized

def show_image(data):
    cv2.namedWindow('First hidden layer weights')
    cv2.imshow('First hidden layer weights', data)

def show_first_hidden_layer(data):
    average_image = extract_average_color(data)
    normalized = normalize_to_greyscale(average_image)
    show_image(normalized)