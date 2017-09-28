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

clf = pickle.load(open('color_classifier.sav', 'rb'))
tree.export_graphviz(clf, out_file='color_graph.dot')

"""
,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True
"""
