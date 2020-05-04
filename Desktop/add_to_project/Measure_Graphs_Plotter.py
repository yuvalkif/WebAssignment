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
import Decision_Tree as dt
import K_Nearest_Neighbors as knn
import Support_Vector_Machine as svm
import IWFS as fs

from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from preprocess_wdbc import data_processing
import pandas as pd
import numpy as np
import os


def get_train_test_split(X, y, cross_validation_split, n_splits=10):
    for train_index, test_index in cross_validation_split:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

'''
TODO finish documentation
     validate auc graph with multiple classes
     validate accuarcy auc graphs with multiple classes
     get target pdf directory name by parameter
'''
class Measure_Graphs_Plotter:
    def __init__(self, feature_selection_technique_name, classifierWrapper, number_of_classes, features_vectors, class_vector):
        self.feature_selection_technique_name = feature_selection_technique_name
        self.classifier = classifierWrapper.get_classifier()
        self.number_of_classes = number_of_classes
        self.features_vectors = features_vectors
        self.class_vector = class_vector
        self.cross_validation = StratifiedKFold(n_splits=10)
        self.cross_validation_split = self.cross_validation.split(self.features_vectors.values.astype(int), self.class_vector.values.astype(int))
        self.X_train, self.X_test, self.Y_train, self.Y_test = get_train_test_split(self.features_vectors.values.astype(int), self.class_vector.values.astype(int),self.cross_validation_split, 10)





    '''
    function for random splitting cross validation results and plotting each split roc and the mean roc of all
    @param save_pdf Boolean to save to pdf file or show the plot
    '''
    def plot_auc_graph(self, save_pdf = False):
        cv = StratifiedKFold(n_splits=10)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        X = self.features_vectors.values.astype(int)
        y = self.class_vector.values.astype(int)
        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(X, y)):
            self.classifier.fit(X[train], y[train])
            viz = plot_roc_curve(self.classifier, X[test], y[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title="ROC graph")
        ax.legend(loc="lower right")
        my_path = os.path.abspath("./measures_graphs")
        my_file = '%s_auc_graph_graph.pdf'%(self.feature_selection_technique_name)
        if save_pdf:
            plt.savefig(os.path.join(my_path, my_file))
        else:
            plt.show()



    '''
        precision recall curve for multi class with f1 measure
        @param save_pdf Boolean to save to pdf file or show the plot
    '''
    def plot_f1_graph(self, save_pdf = False):
        Y = label_binarize(self.class_vector.values.astype(int), classes= np.arange(0,self.number_of_classes))
        n_classes = Y.shape[1]
        print(n_classes)
        X_train, X_test, Y_train, Y_test = self.X_train, self.X_test, self.Y_train, self.Y_test
        Y_train = label_binarize(Y_train, classes= np.arange(0,self.number_of_classes))
        Y_test = label_binarize(Y_test, classes= np.arange(0,self.number_of_classes))
        classifier = OneVsRestClassifier(self.classifier)
        classifier.fit(X_train, Y_train)
        y_score = classifier.predict(X_test)
        if n_classes <= 2:
            y_score = label_binarize(y_score, classes= np.arange(0,self.number_of_classes))
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test[:, i], y_score[:, i],
                                                             average="micro")
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recalls with Fmeasure')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        my_path = os.path.abspath("./measures_graphs")
        my_file = '%s_prec_recall_f1_graph.pdf'%(self.feature_selection_technique_name)
        if save_pdf:
            plt.savefig(os.path.join(my_path, my_file))
        else:
            plt.show()

    '''
    This graph will be used to find hypter parameters from our models e.g gamma for SVM and such
    Accuarcy and AUC graph after Gridsearch cross validation
    @param save_pdf Boolean to save to pdf file or show the plot
    '''
    def plot_accuarcy_auc_graph(self, save_pdf = False):
        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        gs = GridSearchCV(self.classifier,
                          param_grid={'min_samples_split': range(2, 400, 10)},
                          scoring=scoring, refit='AUC', return_train_score=True)
        gs.fit(self.features_vectors, self.class_vector)
        results = gs.cv_results_
        plt.figure(figsize=(13, 13))
        plt.title("AUC and Accuarcy",
                  fontsize=16)

        plt.xlabel("min_samples_split")
        plt.ylabel("Score")

        ax = plt.gca()
        ax.set_xlim(0, 402)
        ax.set_ylim(0.73, 1)

        X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

        for scorer, color in zip(sorted(scoring), ['g', 'k']):
            for sample, style in (('train', '--'), ('test', '-')):
                sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
                sample_score_std = results['std_%s_%s' % (sample, scorer)]
                ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)
                ax.plot(X_axis, sample_score_mean, style, color=color,
                        alpha=1 if sample == 'test' else 0.7,
                        label="%s (%s)" % (scorer, sample))

            best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
            best_score = results['mean_test_%s' % scorer][best_index]

            ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                    linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

            ax.annotate("%0.2f" % best_score,
                        (X_axis[best_index], best_score + 0.005))

        plt.legend(loc="best")
        plt.grid(False)
        my_path = os.path.abspath("./measures_graphs")
        my_file = '%saccuarcy_auc_graph.pdf'%(self.feature_selection_technique_name)
        if save_pdf:
            plt.savefig(os.path.join(my_path, my_file))
        else:
            plt.show()


features_vectors, class_vector, features_names, class_name = data_processing('wdbc.csv', sep=',', number_of_bins=10).prepare_data()
selector = fs.Wrapped_IWFS(features_vectors=features_vectors, class_vector=class_vector, num_of_features=len(features_names), features_names = features_names , class_name = class_name)
selected_features, score = selector.select_features()
classifer = dt.Decision_Tree(features_vectors, class_vector, selected_features , class_name)
# classifer = knn.K_Nearest_Neighbors(features_vectors, class_vector, selected_features , class_name, 3)
# classifer = svm.Support_Vector_Machine(features_vectors, class_vector, selected_features , class_name)

grapher = Measure_Graphs_Plotter("shita lbdika", classifer, 2, features_vectors, class_vector)
grapher.plot_auc_graph()
grapher.plot_f1_graph()
grapher.plot_accuarcy_auc_graph()

