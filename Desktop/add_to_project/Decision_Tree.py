from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from preprocess_wdbc import data_processing
import pandas as pd
from sklearn import tree
import numpy as np
import os


'''
@params
X pandas data frame of all features
Y pandas series class vector
features_names names of the features used to train and classify  
'''
class Decision_Tree:
    def __init__(self, X, Y, features_names, class_name):
        self.features_vectors = X[features_names]
        self.class_vector = Y
        self.features_names = features_names
        self.class_name = class_name
        self.classifier = tree.DecisionTreeClassifier()
        self.feature_selection_technique_name = self.__class__.__name__


    def get_mean_accuarcy_by_cross_validation(self, cross_validations=10):
        cross_val_results = cross_val_score(self.classifier, self.features_vectors[self.features_names].values.astype(int), self.class_vector.values.astype(int), cv=10)
        score = sum(cross_val_results)/len(cross_val_results)
        return score

    def plot_mean_auc_graph(self, save_to_pdf = False):
        self.graph_plotter.plot_auc_graph

    def get_classifier(self):
        return self.classifier



'''
sample run
'''
# features_vectors, class_vector, features_names, class_name = data_processing('wdbc.csv', sep=',', number_of_bins=10).prepare_data()
# model = Decision_Tree(features_vectors, class_vector, ["2","4","17"], class_name)

# print(model.get_model_name())
# print(model.feature_selection_technique_name)
# # score = model.plot_auc_graph()
# model.plot_f1_graph(2, True)
