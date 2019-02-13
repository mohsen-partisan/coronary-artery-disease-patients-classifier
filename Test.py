
import pandas as pd


filename = '/home/mohsen/Desktop/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
print(data.shape)
headers = list(data)
s=data['ECG1ECGThirdDegree'].value_counts()
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName'
                  ], axis=1)
print(data.shape)
print("ha")