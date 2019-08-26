
import pandas as pd
import numpy as np
from MissValueHandler import MissValueHandler
from matplotlib import pyplot as plt
import matplotlib as mpl
from  Correlation import Correlation
from Util import Util


filename = '/home/mohsen/payanname/resources/modifiedDatePrimaryPCI.csv'
data = pd.read_csv(filename, sep=';')
# just needed columns
data = data[data.columns[0:64]]

# create an empty column for target values
data['target'] = np.nan
# a method for creating target values from differentiate between admission date and discharge date
data = Util().compute_hospitalization_length(data)

# removing redundant columns
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'GC1GeneralCharacteristicsInsuranceCo',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName', 'Echo1EchoFindingsGlobalEF', 'CathLabDataCathLabDataStentThrombosis',
                  'D41stLesionPCI1stLesionPCIACCAHAType', 'D41stLesionPCITreatedwithStentStentdiameter',
                  'D41stLesionPCITreatedwithStentStentlenght', 'OtherDataResultsresult',
                  'CathLabDataCathLabDataInitialTIMI', 'CathLabDataCathLabDataFinalTIMI', 'MACE', 'AdmissionPainOnsetDate',
                  'DemographicsDemographicsDateofDischarge'
                  ], axis=1)



column_names = list(data)

# move age column position next to another continuous columns
age_column = data.pop('Age_On_Admission')
data.insert(2, 'Age_On_Admission', age_column)

# using regex to replace blank with 'nan'
data = data.replace(r'^\s*$', np.nan, regex=True)
data = data.replace('.', np.nan)

# create a dictionary of columns that have miss values
columns_with_missing = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        columns_with_missing[data.columns[i]] = sum

# replace nan in numerical columns with '-1'
data = data.replace({'LABDATA1LabDataCR':np.nan}, -1)

# a class for handling miss values
miss_value_handler = MissValueHandler()

data = miss_value_handler.numerical_imputation(data)
data = miss_value_handler.categorical_imputation(data)

# one method to encode categorical values
# data = pd.get_dummies(data, prefix_sep='_')
# one method to encode categorical values
data = Util().categorical_encoder(data)

# showing distribution of hospitalization days
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Days Frequency", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Quality")
ax.set_ylabel("Frequency")
w_q = data['target'].value_counts()
w_q = (list(w_q.index), list(w_q.values))
ax.tick_params(axis='both', which='major', labelsize=8.5)
bar = ax.bar(w_q[0], w_q[1], color='steelblue')
plt.show()


# creating three data['target'] < 4, 'target'] = 0
# data.loc[data['target'] == 4, 'target'] = 1
# data.loc[data['target'] == 5, 'target'] = 1
# data.loc[data['target'] >= 6, 'target'] = 2

data.loc[data['target'] < 6,'target'] = 0
data.loc[data['target'] >= 6,'target'] = 1

value_counts = data['target'].value_counts()


# Correlation().calculate_correaltion(data)
data.dtypes
print(data.shape)
# return data to apply featureSelection
def getData():
    return data




