"""
ColorRecognizer
===============
Navigation:
	esc - quit the app
	
Annotation:
	r - save as red color
	g - save as green color
	b - save as blue color
	o - save as orange color
	y - save as yellow color
	v - save as violet color
	k - save as black color
	w - save as white color
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn import tree
import pickle
from numpy import genfromtxt

w = 448
h = 252
s = 200

labels = ['red', 'green', 'blue', 'orange', 'yellow', 'violet', 'black', 'white']
current_color = 'None'

def save_data(data):
	np.save('color_dataset.npz', data)
	print('Dataset was saved into numpy binary')

def save_classifier(clf):
	pickle.dump(clf, open('color_classifier.sav', 'wb'))
	print('Classifier was saved!')
	
def draw_text(img, text, color):	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, text, (int(w/20), int(h/2)), font, 1, color, 2, cv2.LINE_AA)

def show_webcam():
	# Load classifier
	classifier = Path('color_classifier.sav')
	if classifier.is_file():
		clf = pickle.load(open('color_classifier.sav', 'rb'))
		print('Classifier was loaded!')
		print(clf)
		
		data = genfromtxt('color_dataset.csv', delimiter=',')
		print('Learning set loaded')
		
		X = data[:, [0,1,2]]
		y = data[:, 3]
		clf = clf.fit(X, y)
		print('Classifier updated from last session!!!\n')
	else:
		print('No saved classifier I will create a new one!')
		clf = tree.DecisionTreeClassifier()
	f = open('color_dataset.csv','ab')
	
	cam = cv2.VideoCapture(0)
	while True:
		ret_val, img = cam.read()
		img = cv2.flip(img, 1)
		img = cv2.resize(img, (0, 0), fx=0.35, fy=0.35)

		# OpenCV read color in BGR
		average_color_per_row = np.average(img, axis=0)
		average_color = np.average(average_color_per_row, axis=0)
		average_color = np.uint8(average_color)
		
		r = average_color[2].tolist()
		g = average_color[1].tolist()
		b = average_color[0].tolist()
		
		data = np.array([r, g, b], np.int).reshape(1, 3)
		
		# Predict the color based on updated classifier
		p = clf.predict(data)
		print(labels[int(p)])
		current_color = labels[int(p)]
		draw_text(img, current_color, (0, 255, 255))
		
		cv2.imshow('ColorRecognizer', img)
		cv2.moveWindow('ColorRecognizer', 0, 0)
		key = cv2.waitKey(1)
		if key == ord('r'):
			np.savetxt(f, np.append(data, 0).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as red')
		elif key == ord('g'):
			np.savetxt(f, np.append(data, 1).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as green')
		elif key == ord('b'):
			np.savetxt(f, np.append(data, 2).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as blue')
		elif key == ord('o'):
			np.savetxt(f, np.append(data, 3).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as orange')
		elif key == ord('y'):
			np.savetxt(f, np.append(data, 4).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as yellow')
		elif key == ord('v'):
			np.savetxt(f, np.append(data, 5).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as violet')
		elif key == ord('k'):
			np.savetxt(f, np.append(data, 6).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as black')
		elif key == ord('w'):
			np.savetxt(f, np.append(data, 7).reshape(1,4), fmt='%d', delimiter=',')
			print('Annotated as white')
		elif key == 27:
			print('Closing the app...')
			break
	f.close()
	save_classifier(clf)
	cv2.destroyAllWindows()

def main():
	show_webcam()

if __name__ == '__main__':
	main()