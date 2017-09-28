# Color Recognizer

This program is written for learning purpose of
building the naturally inteligent system, e.g.
cognitive architecture.

Goal is to classify the colors based on any neural network
algorhytm with web camera input.

## How it works

Program reads a web camera and analyze the images in real
time while user can annotate images also in real time.

For annotation one has to place a object in front of camera.
Best practice is to place object as close as possible to gain
a flat color in camera window. The it is computed average RGB
value from the image and a) predict the color or b) user can
annotate the color by him/her self by pressing the keyboard as follows:


| Color | Shortcut |
|-------|----------|
| White | w        |
| Black | k        |
| Red   | r        |
| Green | g        |
| Blue  | b        |
| Orange | o       |
| Purple | p       |
| Violet | v       |


For quit the program just press ESC key while focused on main
window.

## Neuronal network

It was used a Decision Tree Classifier for training neuronal
network because of it is easy interpretation and exploring
the meaning of the classification at the begining.

This is the setup (actually the basic one):
```python
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
```

Each session is saved and on the start of another one classifier
is trained again to strength the classification.

This is link to [documentation page](http://scikit-learn.org/stable/modules/tree.html) to Decision Tree Classifier in Scikit-learn library.

## Dependecies

- Python 3.6
- OpenCV 2
- NumPy
- Scikit-learn
- Web camera (obviously)

## Running

Open the code ```color_recognizer.py``` in any Python IDE
or editor and simply run it. Or you can navigate by terminal
into direction of file and run it by ```python3 color_recognizer.py```

# Demo

This is a real demo of early training (e.g., 50-60 per training images per color)

![https://github.com/neuropacabra/ColorRecognizer/blob/doc/cr_demo_training_01.gif?raw=true](https://github.com/neuropacabra/ColorRecognizer/blob/doc/cr_demo_training_01.gif?raw=true)
