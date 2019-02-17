
import pandas as pd
from sklearn import preprocessing
import numpy as np

filename = '/home/mohsen/Desktop/pci/new/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'GC1GeneralCharacteristicsInsuranceCo',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName'
                  ], axis=1)
headers = list(data)
data = data.replace(r'^\s*$', np.nan, regex=True)
data = data.replace('.', np.nan)
s=data['MACE'].value_counts()
miss = data['MACE'].isnull().sum()

# encode categorical values
# le = preprocessing.LabelEncoder()
# for i in range(0,data.shape[1]):
#     if data.dtypes[i]=='object':
#         data[data.columns[i]] = le.fit_transform(data[data.columns[i]])
print("ha")