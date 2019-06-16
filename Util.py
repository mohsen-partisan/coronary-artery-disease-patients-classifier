
import pandas as pd
from jdatetime import date
import sklearn.preprocessing as preprocessing

class Util:

     def selected_features(self, data):
        non_zero = []
        i=0
        for item in data:
            if item !=0:
                non_zero.append(data.index(item))
                i = i+1
        return non_zero

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
