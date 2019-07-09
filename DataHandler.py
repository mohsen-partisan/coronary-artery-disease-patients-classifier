
import pandas as pd
import numpy as np
from MissValueHandler import MissValueHandler
from Util import Util
filename = '/home/mohsen/payanname/resources/modifiedDatePrimaryPCI.csv'
data = pd.read_csv(filename, sep=';')
# just needed columns
data = data[data.columns[0:64]]
# removing redundant columns
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'GC1GeneralCharacteristicsInsuranceCo',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName', 'Echo1EchoFindingsGlobalEF'
                  ], axis=1)


column_names = list(data)
# using regex to replace blank with 'nan'
data = data.replace(r'^\s*$', np.nan, regex=True)
data = data.replace('.', np.nan)
# replace nan in numerical columns with '-1'
data = data.replace({'LABDATA1LabDataCR':np.nan, 'LABDATA1LabDataHb':np.nan,
            'D41stLesionPCITreatedwithStentStentdiameter':np.nan,
            'D41stLesionPCITreatedwithStentStentlenght':np.nan}, -1)


# create a dictionary of columns that have miss values
columns_with_missing = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        columns_with_missing[data.columns[i]] = sum

# a class for handling miss values
miss_value_handler = MissValueHandler()

data = miss_value_handler.numerical_imputation(data)
data = miss_value_handler.categorical_imputation(data)

# create a dictionary of columns that have miss values yet(not imputed in previous step)
remain_columns_with_missing = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        remain_columns_with_missing[data.columns[i]] = sum

# create an 'Unknown' category for handling missing values(another method to handle categorical miss values)
data = miss_value_handler.create_new_category(data)

# handling miss values in two dependent columns(following method is not standard.)
data = miss_value_handler.handle_miss_in_dependent_values(data)

# convert data types
convert_dict = {'CathLabDataCathLabDataInitialTIMI': 'int64',
                'CathLabDataCathLabDataFinalTIMI':'int64'}
data = data.astype(convert_dict)

# create an empty column for target values
data['target'] = np.nan

# a method for creating target values from differentiate between admission date and discharge date
data = Util().compute_hospitalization_length(data)

# removing remaining unused columns for creating model
data = data.drop(['AdmissionPainOnsetDate', 'DemographicsDemographicsDateofDischarge', 'OtherDataResultsresult'], axis=1)

# one method to encode categorical values
# data = pd.get_dummies(data, prefix_sep='_')

# one method to encode categorical values
data = Util().categorical_encoder(data)
# creating two classes in target according to problem requirement
data.loc[data['target'] < 6, 'target'] = 0
data.loc[data['target'] >= 6, 'target'] = 1
# move target to last column
data = data[[c for c in data if c not in ['target']] + ['target']]

# return data to apply featureSelection
def getData():
    return data




