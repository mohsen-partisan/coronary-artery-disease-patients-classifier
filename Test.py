
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

filename = '/home/mohsen/Desktop/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  'GC1GeneralCharacteristicsInsuranceCo',
                  'InitialReperfusionTherapyTransferToCathlabRescuePCI',
                  'ECG1ECGThirdDegree', 'GC1GeneralCharacteristicsAdmision',
                  'PatientFullName', 'Echo1EchoFindingsGlobalEF'
                  ], axis=1)
headers = list(data)
data = data.replace(r'^\s*$', np.nan, regex=True)
data = data.replace('.', np.nan)
data = data.replace({'LABDATA1LabDataCR':np.nan, 'LABDATA1LabDataHb':np.nan,
            'D41stLesionPCITreatedwithStentStentdiameter':np.nan,
            'D41stLesionPCITreatedwithStentStentlenght':np.nan}, -1)
s=data['CathLabDataCathLabDataInitialTIMI'].value_counts()
miss = data['CathLabDataCathLabDataStentThrombosis'].isnull().sum()
nanColumns = data.columns[data.isnull().any()].tolist()
dict = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        dict[data.columns[i]] = sum

integer_imp = SimpleImputer(missing_values=-1, strategy='mean', fill_value=0, copy=False)
categorical_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

data[['D41stLesionPCITreatedwithStentStentdiameter','LABDATA1LabDataHb',
      'LABDATA1LabDataCR', 'D41stLesionPCITreatedwithStentStentlenght']] = integer_imp.fit_transform(data[['D41stLesionPCITreatedwithStentStentdiameter','LABDATA1LabDataHb',
      'LABDATA1LabDataCR', 'D41stLesionPCITreatedwithStentStentlenght']])

data[['PMH1PastMedicalHistrySuccessfulCPR', 'CADRF1CADRFOpium',
      'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
      'ECG1ECGPVC', 'GC1GeneralCharacteristicsMaritalStatus',
      'PMH1PastMedicalHistryCardiomyopathy', 'PMH1PastMedicalHistryChronicLungDisease',
      'PMH1PastMedicalHistryDialysis']] = categorical_imp.fit_transform(data[['PMH1PastMedicalHistrySuccessfulCPR', 'CADRF1CADRFOpium',
      'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
      'ECG1ECGPVC', 'GC1GeneralCharacteristicsMaritalStatus',
      'PMH1PastMedicalHistryCardiomyopathy', 'PMH1PastMedicalHistryChronicLungDisease',
      'PMH1PastMedicalHistryDialysis']])

dict2 = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        dict2[data.columns[i]] = sum
data = data.replace({'CathLabDataCathLabDataStentThrombosis':np.nan, 'MACE':np.nan,
                     'D41stLesionPCI1stLesionPCIACCAHAType':np.nan,
                     'OtherDataResultsresult':np.nan}, 'Unknown')

int_timi_nan = data['CathLabDataCathLabDataInitialTIMI'].isnull()
count = 0
sum_timi = 0
for i in range(0, data.shape[0]):
    if not int_timi_nan[i]:
      sum_timi += int(data.iloc[i]['CathLabDataCathLabDataFinalTIMI']) - int(data.iloc[i]['CathLabDataCathLabDataInitialTIMI'])
      count+=1
avg = sum / count
# for i in range(0, data.shape[0]):
# if data[data.columns[0]]
# encode categorical values
# le = preprocessing.LabelEncoder()
# for i in range(0,data.shape[1]):
#     if data.dtypes[i]=='object':
#         data[data.columns[i]] = le.fit_transform(data[data.columns[i]])
print("ha")