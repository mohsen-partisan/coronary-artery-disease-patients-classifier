
import pandas as pd
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as preprocessing
import numpy as np
from jdatetime import date
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
      count+=1
avg = sum_timi / count
inti_timi_imputer = SimpleImputer(strategy='most_frequent')
data['CathLabDataCathLabDataInitialTIMI'] = inti_timi_imputer.fit_transform(data[['CathLabDataCathLabDataInitialTIMI']])
data = data.replace({'CathLabDataCathLabDataFinalTIMI':np.nan}, avg)
data['target'] = np.nan


def split_date(date):
    return date.split('/')
in_date = 0
out_date = 0
for i in range(0, (data.shape[0])):
    in_date = data.iloc[i]['AdmissionPainOnsetDate']
    out_date = data.iloc[i]['DemographicsDemographicsDateofDischarge']
    splitted_in_date = split_date(in_date)
    splitted_out_date = split_date(out_date)
    d1 = date(int(splitted_in_date[0]), int(splitted_in_date[1]), int(splitted_in_date[2]))
    d2 = date(int(splitted_out_date[0]), int(splitted_out_date[1]), int(splitted_out_date[2]))
    data.loc[i, 'target'] = (d2 - d1).days

long = data[data.target > 10]
types = data.dtypes
data = data.drop(['AdmissionPainOnsetDate', 'DemographicsDemographicsDateofDischarge'], axis=1)
data = pd.get_dummies(data, prefix_sep='_')

# for i in range(0, data.shape[0]):
# if data[data.columns[0]]
# encode categorical values
# le = preprocessing.LabelEncoder()
# for i in range(0,data.shape[1]):
#     if data.dtypes[i]=='object':
#         data[data.columns[i]] = le.fit_transform(data[data.columns[i]])
print("ha")