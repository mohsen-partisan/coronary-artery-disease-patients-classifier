
import pandas as pd
import collections
from jdatetime import date
import sklearn.preprocessing as preprocessing
from scipy.spatial import distance_matrix
from sklearn.metrics import classification_report, accuracy_score, make_scorer, confusion_matrix
class Util:

     def selected_features_for_xgboost(self, features):
         d = dict((item, features.index(item)) for item in features)
         od = collections.OrderedDict(sorted(d.items(), reverse=True))
         values = []
         for k, v in od.items():
             values.append(v)
         return values[:]

     def compute_hospitalization_length(self, data):
         in_date = 0
         out_date = 0
         for i in range(0, (data.shape[0])):
             in_date = data.iloc[i]['AdmissionPainOnsetDate']
             out_date = data.iloc[i]['DemographicsDemographicsDateofDischarge']
             splitted_in_date = self.split_date(in_date)
             splitted_out_date = self.split_date(out_date)
             d1 = date(int(splitted_in_date[0]), int(splitted_in_date[1]), int(splitted_in_date[2]))
             d2 = date(int(splitted_out_date[0]), int(splitted_out_date[1]), int(splitted_out_date[2]))
             data.loc[i, 'target'] = (d2 - d1).days

         return data

     def split_date(self, date):
         return date.split('/')

     def categorical_encoder(self, data):
         label_encoder = preprocessing.LabelEncoder()
         for i in range(0, data.shape[1]):
             if data.dtypes[i] == 'object':
                 data[data.columns[i]] = label_encoder.fit_transform(data[data.columns[i]])

         return data

     def classification_report_with_accuracy_score(target_test, predictions):
         print(classification_report(target_test, predictions))  # print classification report
         print(confusion_matrix(target_test, predictions, labels=[0, 1]))
         print(accuracy_score(target_test, predictions))
         return accuracy_score(target_test, predictions)  # return accuracy score

     def distance_matrix(self, data):
        sorted_data = data.sort_values('target')
        target = sorted_data.target
        sorted_data = sorted_data.drop(['target'], axis=1)
        distance_mat = pd.DataFrame(distance_matrix(sorted_data.values, sorted_data.values), index=sorted_data.index,
                                    columns=sorted_data.index)
        distance_mat['target'] = target
        return distance_mat
