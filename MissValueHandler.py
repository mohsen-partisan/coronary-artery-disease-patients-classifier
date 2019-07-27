
from sklearn.impute import SimpleImputer
import numpy as np

class MissValueHandler:

    categorical_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    def numerical_imputation(self, data):
        # imputation of numerical columns
        integer_imp = SimpleImputer(missing_values=-1, strategy='mean', fill_value=0, copy=False)
        data[['LABDATA1LabDataCR']] = integer_imp.fit_transform(
            data[['LABDATA1LabDataCR']])
        return  data

    def categorical_imputation(self, data):
        # imputation of some of categorical columns with 'most frequent' method
        data[['PMH1PastMedicalHistrySuccessfulCPR',
              'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
              'GC1GeneralCharacteristicsMaritalStatus']] = self.categorical_imp.fit_transform(
            data[['PMH1PastMedicalHistrySuccessfulCPR',
                  'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
                  'GC1GeneralCharacteristicsMaritalStatus']])
        return data

    # a custom method for handle miss in two dependent columns
    def handle_miss_in_dependent_values(self, data):
        int_timi_nan = data['CathLabDataCathLabDataInitialTIMI'].isnull()
        fin_timi_nan = data['CathLabDataCathLabDataFinalTIMI'].isnull()
        count = 0
        sum_timi = 0
        a = 0
        b = 0
        for i in range(0, (data.shape[0])):
            if not int_timi_nan[i] and not fin_timi_nan[i]:
                a = int(data.iloc[i]['CathLabDataCathLabDataFinalTIMI'])
                b = int(data.iloc[i]['CathLabDataCathLabDataInitialTIMI'])
                sum_timi += a - b
                count += 1
        avg = sum_timi / count

        data['CathLabDataCathLabDataInitialTIMI'] = self.categorical_imp.fit_transform(
            data[['CathLabDataCathLabDataInitialTIMI']])
        # apply heuristic method for handling miss values in 'CathLabDataCathLabDataFinalTIMI'
        data = data.replace({'CathLabDataCathLabDataFinalTIMI': np.nan}, avg)

        return data

    # change 'nan' values to 'Uknown' to be a new category
    def create_new_category(self, data):
        data = data.replace({'CathLabDataCathLabDataStentThrombosis': np.nan, 'MACE': np.nan,
                             'D41stLesionPCI1stLesionPCIACCAHAType': np.nan,
                             'OtherDataResultsresult': np.nan}, 'Unknown')
        return  data
