
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from FeatureSelection import FeatureSelection
from Util import Util
import pandas as pd
from DataHandler import getData

class DataPreprocessor:

    def __init__(self):
        self.data = getData()

    def split_out_train_test(self):
        standardized_array = self.standardize()
        # extract target from array
        target = standardized_array[:, (standardized_array.shape[1] - 1)]
        # extract selected features
        features = self.select_features()
        validation_size = 0.20
        seed = 7
        return train_test_split(features, target, test_size=validation_size, random_state=seed)


    def standardize(self):
        array = self.data.values

        # standardize continuous columns
        scaler = StandardScaler().fit(array[:, 0:3])
        array[:, 0:3] = scaler.transform(array[:, 0:3])
        return array

    def normalize(self):
        array = self.data.values
        # normalize continuous columns
        array[:, 0:3] = normalize(array[:, 0:3])
        return array


    # feature selection
    def select_features(self):
        feature_selection = FeatureSelection()

        # select features using xgboost
        selected_features = feature_selection.selected_features_by_xgboost(self.normalize())
        selected_features_index = Util().selected_features_for_xgboost(selected_features)
        selected_features_by_xgboost = self.data.ix[:, selected_features_index]
        features_with_target = pd.concat([selected_features_by_xgboost, self.data['target']], axis=1)
        return features_with_target

        # selected_features = feature_selection.selected_features_by_selectKBest(self.normalize())
        # features_with_target = pd.concat([pd.DataFrame(data=selected_features), self.data['target']], axis=1)
        # return features_with_target

    def select_all_features(self):
        array = self.standardize()
        column_names = list(self.data)
        return pd.DataFrame(array,columns=column_names)