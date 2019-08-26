
import numpy as np
import random
from sklearn.cluster import KMeans
from DataPreprocessor import DataPreprocessor
from DataHandler import getData
from yellowbrick.cluster import KElbowVisualizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from DataHandler import getData
import pandas as pd

class Clustering:

    def __init__(self):
        self.train_index = []
        self.test_index = []
        self.data = pd.DataFrame()

    def cluster_data_before_classify(self, data):
        model = KMeans(n_clusters=2, random_state=8)

        # visualizer = KElbowVisualizer(model, k=(2, 8), metric='silhouette', timings=False)
        # # Fit the data and visualize
        # visualizer.fit(data)
        # visualizer.poof()
        model.fit(data)

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

        model = RandomForestClassifier()
        self.train_model(data_train, model)

        result = model.score(data_test, target_test)
        predicted = model.predict(data_test)
        report = classification_report(target_test, predicted)
        matrix = confusion_matrix(target_test, predicted)
        print(("Accuracy for test: %.3f%%") % (result * 100.0))
        print(report)
        print(matrix)

    def train_model(self, data_train, model):
        target_train = data_train.target
        data_train = data_train.drop(['target'], axis=1)
        trained_model = model.fit(data_train, target_train)
        result_train = accuracy_score(target_train, trained_model.predict(data_train))
        matrix_train = confusion_matrix(target_train, trained_model.predict(data_train))
        print(("Accuracy for train: %.3f%%") % (result_train * 100.0))
        print(matrix_train)



    def get_features(self):
        self.data = DataPreprocessor().select_features()
        # self.data = getData()
        self.cluster_data_before_classify(self.data)


Clustering().get_features()