
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy

class FeatureSelection:

    def selected_features_by_xgboost(self, data):
        features = data[:, 0:(data.shape[1]-1)]
        target = data[:, (data.shape[1]-1)]
        xmodel = XGBClassifier()
        xmodel.fit(features, target)
        return list(xmodel.feature_importances_)

    def selected_features_by_selectKBest(self, data):
        features = data[:, 0:(data.shape[1] - 1)]
        target = data[:, (data.shape[1] - 1)]

        best_features = SelectKBest(score_func=chi2, k=20)
        fitted_best_features = best_features.fit(features, target)
        numpy.set_printoptions(precision=3)
        print(fitted_best_features.scores_)
        selected_features = fitted_best_features.transform(features)
        return selected_features

    def selected_features_by_recursice_feature_elimination(self, data):
        features = data[:, 0:(data.shape[1] - 1)]
        target = data[:, (data.shape[1] - 1)]


    def selected_features_by_pca(self, data):
        features = data[:, 0:(data.shape[1] - 1)]
        target = data[:, (data.shape[1] - 1)]


    def selected_features_by_extra_tree_classifier(self, data):
        features = data[:, 0:(data.shape[1] - 1)]
        target = data[:, (data.shape[1] - 1)]
