"""
Export trained classifier to graph
==================================
Load trained classifier and save the whole model
as tree graph (true/false) with all features and
target colors. It assumes that classifier is
already generated and saved.

Usage:
    dot -Tpng color_graph.dot -o tree.png

Install dot command line tool:
    xcode-select --install

Michael Tesar <michtesar@gmail.com>
28-9-2017 Drahotesice
"""

import pickle
from sklearn import tree

LABELS = ['red', 'green', 'blue', 'orange',
          'yellow', 'violet', 'black', 'white']

clf = pickle.load(open('color_classifier.sav', 'rb'))
tree.export_graphviz(clf, out_file='color_graph.dot',
                     feature_names=['Red', 'Green', 'Blue'],
                     rounded=True, class_names=LABELS,
                     filled=True, max_depth=5, proportion=True)
