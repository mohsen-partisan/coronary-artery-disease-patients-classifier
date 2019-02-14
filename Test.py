
import pandas as pd
from sklearn import preprocessing


filename = '/home/mohsen/Desktop/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
headers = list(data)
s=data['D41stLesionPCI1stLesionPCIACCAHAType'].value_counts()
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'GC1GeneralCharacteristicsInsuranceCo',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName'
                  ], axis=1)
data.dtypes
for i in data['D41stLesionPCI1stLesionPCIACCAHAType']:
    if i.empty:
        data['D41stLesionPCI1stLesionPCIACCAHAType'] = 'NaN'
# encode categorical values
le = preprocessing.LabelEncoder()
for i in range(0,data.shape[1]):
    if data.dtypes[i]=='object':
        data[data.columns[i]] = le.fit_transform(data[data.columns[i]])
print("ha")