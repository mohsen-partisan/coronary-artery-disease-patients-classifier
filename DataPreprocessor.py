
from sklearn.preprocessing import StandardScaler
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
        # move age column position next to another continious columns
        age_column = self.data.pop('Age_On_Admission')
        self.data.insert(2, 'Age_On_Admission', age_column)
        self.final_headers = list(self.data)
        array = self.data.values

        # standardize continious columns
        scaler = StandardScaler().fit(array[:, 0:3])
        array[:, 0:3] = scaler.transform(array[:, 0:3])

        return array

    # feature selection
    def select_features(self):
        feature_selection = FeatureSelection()

        # select features using xgboost
        selected_features = feature_selection.selected_features_by_xgboost(self.standardize())
        selected_features_index = Util().selected_features_for_xgboost(selected_features)
        selected_features_by_xgboost = self.data.ix[:, selected_features_index]
        features_with_target = pd.concat([selected_features_by_xgboost, self.data['target']], axis=1)
        return features_with_target

        # select features using selectKBest
        # selected_features_by_kbest = feature_selection.selected_features_by_selectKBest(self.data.values)
        # return selected_features_by_kbest