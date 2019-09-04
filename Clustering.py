
import numpy as np
import random
from sklearn.cluster import KMeans
from kmodes import kprototypes
from DataPreprocessor import DataPreprocessor
from DataHandler import getData
from yellowbrick.cluster import KElbowVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OutputCodeClassifier
from DataHandler import getData
import pandas as pd

class Clustering:

    def __init__(self):
        self.train_index = []
        self.test_index = []
        self.data = pd.DataFrame()

    def cluster_data_before_classify(self, data):
        model = kprototypes.KPrototypes(n_clusters=8)
        cluster_data = data.drop(['target'], axis=1)
        categoricals = [i for i in range(3,len(cluster_data.columns))]

        # visualizer = KElbowVisualizer(model, k=(2, 8), metric='silhouette', timings=False)
        # # Fit the data and visualize
        # visualizer.fit(data)
        # visualizer.poof()
        model.fit(cluster_data.values, categorical=categoricals)

        labels = model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        dict(zip(unique, counts))

        indices = [np.where(model.labels_ == i)[0] for i in range(model.n_clusters)]
        tt = self.data.ix[indices[0]]
        bb = self.data.ix[indices[1]]
        for i in range(model.n_clusters):
            random.shuffle(indices[i])
        self.merge_train_test_data_from_each_cluster(indices)


        a = 0

    def merge_train_test_data_from_each_cluster(self, indices):
        for i in range(len(indices)):
            each_cluster_index = indices[i]
            self.train_index = np.concatenate((self.train_index, each_cluster_index[0:int(len(each_cluster_index)*0.8)]))
            self.test_index = np.concatenate((self.test_index, each_cluster_index[int(len(each_cluster_index)*0.8):len(each_cluster_index)]))
        self.split_dataframe_by_train_test_index()



    def split_dataframe_by_train_test_index(self):
        data_train = self.data.ix[self.train_index]
        data_test =  self.data.ix[self.test_index]

        self.create_classification_model(data_train, data_test)


    def create_classification_model(self, data_train, data_test):
        target_test = data_test.target
        data_test = data_test.drop(['target'], axis=1)

        model = LogisticRegression(class_weight='balanced')
        self.train_model(data_train, model)

        result = model.score(data_test, target_test)
        predicted = model.predict(data_test)
        report = classification_report(target_test, predicted)
        matrix = confusion_matrix(target_test, predicted)
        print(("Accuracy for test: %.3f%%") % (result * 100.0))
        print(report)
        print(matrix)

    def svc_param_tuning(self, X, y, nfolds):
        Cs = [0.00001,0.0001,0.001, 0.01, 0.1, 1, 10, 50, 100]
        gammas = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'C': Cs, 'gamma': gammas}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        return grid_search.best_params_

    def train_model(self, data_train, model):
        target_train = data_train.target
        data_train = data_train.drop(['target'], axis=1)
        # self.svc_param_tuning(data_train, target_train, 10)
        trained_model = model.fit(data_train, target_train)
        result_train = accuracy_score(target_train, trained_model.predict(data_train))
        matrix_train = confusion_matrix(target_train, trained_model.predict(data_train))
        print(("Accuracy for train: %.3f%%") % (result_train * 100.0))
        print(matrix_train)



    def get_features(self):
        self.data = DataPreprocessor().select_all_features()
        # self.data = getData()
        self.cluster_data_before_classify(self.data)


Clustering().get_features()