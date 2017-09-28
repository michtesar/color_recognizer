import pickle
from sklearn import tree
import graphviz

clf = pickle.load(open('color_classifier.sav', 'rb'))
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph