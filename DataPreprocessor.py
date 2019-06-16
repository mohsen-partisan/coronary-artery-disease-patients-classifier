
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from FeatureSelection import FeatureSelection
from Util import Util
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
        x = self.data.pop('D41stLesionPCITreatedwithStentStentdiameter')
        y = self.data.pop('D41stLesionPCITreatedwithStentStentlenght')
        self.data.insert(2, 'Age_On_Admission', age_column)
        self.data.insert(3, 'D41stLesionPCITreatedwithStentStentdiameter', x)
        self.data.insert(4, 'D41stLesionPCITreatedwithStentStentlenght', y)
        self.final_headers = list(self.data)
        array = self.data.values

        # standardize continious columns
        scaler = StandardScaler().fit(array[:, 0:5])
        array[:, 0:5] = scaler.transform(array[:, 0:5])

        return array

    # feature selection
    def select_features(self):
        feature_selection = FeatureSelection()
        selected_features = feature_selection.selected_features_by_xgboost(self.standardize())
        selected_features_index = Util().selected_features(selected_features)
        selected_data = self.data.ix[:, selected_features_index]
        return selected_data.values