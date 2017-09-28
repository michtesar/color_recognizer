"""
Color recognizer for training neuronal network
==============================================
Learning of naturally insteligent systems by
classifying the color based on web camera
input by Decision Tree Classifier.

Michael Tesar <michtesar@gmail.com>
28-09-2017 Prague

Version 1.2.0
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn import tree

LABELS = ['red', 'green', 'blue', 'orange',
          'yellow', 'violet', 'black', 'white']


def save_classifier(clf):
    """
    Save classifier for futher use in another
    instance of runtime
    Input:
        clf (Scikit-learn structure) - trained
        classificator
    """
    pickle.dump(clf, open('color_classifier.sav', 'wb'))
    print('Classifier was saved!')


def draw_text(img, text, color, res):
    """
    Draws a color name in camera window
    Input:
        img (ndarray) - image data from camera
        text (string) - name of the color
        color (tupple) - RGB color of the text
        res (array) - image resolution
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img, text, (int(res[1] / 20), int(res[0] / 2)),
        font, 1, color, 2, cv2.LINE_AA)


def show_webcam():
    """
    Show little GUI to the user where it can
    be annotated a camera input or/and predict
    the input image in real time
    """
    # Load classifier and train it for new session
    classifier = Path('color_classifier.sav')
    if classifier.is_file():
        clf = pickle.load(open('color_classifier.sav', 'rb'))
        print('Classifier was loaded!')
        print(clf)

        # Load previously saved training dataset
        data = np.genfromtxt('color_dataset.csv', delimiter=',')
        print('Learning set loaded')

        # Shape the data and classify
        X = data[:, [0, 1, 2]]
        y = data[:, 3]
        clf = clf.fit(X, y)
        print('Classifier updated from last session!!!\n')
    else:
        print('No saved classifier found I will create a new one!')
        clf = tree.DecisionTreeClassifier()

    # Open dataset for appending each annotation
    f = open('color_dataset.csv', 'ab')

    # Create a instance of camera object
    cam = cv2.VideoCapture(0)
    while True:
        # Read data from camera into buffer
        _, img = cam.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (0, 0), fx=0.35, fy=0.35)

        # Average color of the image into RGB
        average_color_per_row = np.average(img, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8(average_color)

        # OpenCV stores data in BGR order
        r = average_color[2].tolist()
        g = average_color[1].tolist()
        b = average_color[0].tolist()

        # Final RGB data for training or predicting
        data = np.array([r, g, b], np.int).reshape(1, 3)

        # Predict the color based on updated classifier
        p = clf.predict(data)
        print(LABELS[int(p)])
        current_color = LABELS[int(p)]

        # Draw color name in image
        draw_text(img, current_color, (0, 255, 255), np.shape(img))

        # Flip the screen
        cv2.imshow('ColorRecognizer', img)

        # Move it into corner to be more reachable in macOS
        cv2.moveWindow('ColorRecognizer', 0, 0)

        # Collect the keyboard response
        key = cv2.waitKey(1)
        if key == ord('r'):
            np.savetxt(f, np.append(data, 0).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as red')
        elif key == ord('g'):
            np.savetxt(f, np.append(data, 1).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as green')
        elif key == ord('b'):
            np.savetxt(f, np.append(data, 2).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as blue')
        elif key == ord('o'):
            np.savetxt(f, np.append(data, 3).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as orange')
        elif key == ord('y'):
            np.savetxt(f, np.append(data, 4).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as yellow')
        elif key == ord('v'):
            np.savetxt(f, np.append(data, 5).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as violet')
        elif key == ord('k'):
            np.savetxt(f, np.append(data, 6).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as black')
        elif key == ord('w'):
            np.savetxt(f, np.append(data, 7).reshape(
                1, 4), fmt='%d', delimiter=',')
            print('Annotated as white')
        elif key == 27:
            print('Closing the app...')
            break

    # Uninitialize the file, classifier and window
    f.close()
    save_classifier(clf)
    cv2.destroyAllWindows()


def main():
    """
    Recognize the colors based on previous
    training of classificator and then simply
    predicting the averaged RGB values from
    web camera
    """
    show_webcam()

if __name__ == '__main__':
    main()
