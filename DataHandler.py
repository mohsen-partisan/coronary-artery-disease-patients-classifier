
import pandas as pd
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as preprocessing
import numpy as np
from jdatetime import date
filename = '/home/mohsen/PycharmProjects/payanname/resources/modifiedDatePrimaryPCI.csv'
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
# s=data['CathLabDataCathLabDataInitialTIMI'].value_counts()
# miss = data['CathLabDataCathLabDataStentThrombosis'].isnull().sum()
# nanColumns = data.columns[data.isnull().any()].tolist()

# create a dictionary of columns that have miss values
columns_with_missing = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        columns_with_missing[data.columns[i]] = sum
# imputation of numerical columns
integer_imp = SimpleImputer(missing_values=-1, strategy='mean', fill_value=0, copy=False)
data[['D41stLesionPCITreatedwithStentStentdiameter','LABDATA1LabDataHb',
      'LABDATA1LabDataCR', 'D41stLesionPCITreatedwithStentStentlenght']] = integer_imp.fit_transform(data[['D41stLesionPCITreatedwithStentStentdiameter','LABDATA1LabDataHb',
      'LABDATA1LabDataCR', 'D41stLesionPCITreatedwithStentStentlenght']])

# imputation of some of categorical columns with 'most frequent' method
categorical_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data[['PMH1PastMedicalHistrySuccessfulCPR', 'CADRF1CADRFOpium',
      'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
      'ECG1ECGPVC', 'GC1GeneralCharacteristicsMaritalStatus',
      'PMH1PastMedicalHistryCardiomyopathy', 'PMH1PastMedicalHistryChronicLungDisease',
      'PMH1PastMedicalHistryDialysis']] = categorical_imp.fit_transform(data[['PMH1PastMedicalHistrySuccessfulCPR', 'CADRF1CADRFOpium',
      'GC1GeneralCharacteristicsEducation', 'GC1GeneralCharacteristicsOccupation',
      'ECG1ECGPVC', 'GC1GeneralCharacteristicsMaritalStatus',
      'PMH1PastMedicalHistryCardiomyopathy', 'PMH1PastMedicalHistryChronicLungDisease',
      'PMH1PastMedicalHistryDialysis']])

# create a dictionary of columns that have miss values yet(not imputed in previous step)
remain_columns_with_missing = {}
for i in range(0, data.shape[1]):
    sum = data[data.columns[i]].isnull().sum()
    if sum > 0:
        remain_columns_with_missing[data.columns[i]] = sum

# create an 'Unknown' category for handling missing values(another method to handle categorical miss values)
data = data.replace({'CathLabDataCathLabDataStentThrombosis':np.nan, 'MACE':np.nan,
                     'D41stLesionPCI1stLesionPCIACCAHAType':np.nan,
                     'OtherDataResultsresult':np.nan}, 'Unknown')

# handling miss values in two dependent columns(following method is not standard.)
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

data['CathLabDataCathLabDataInitialTIMI'] = categorical_imp.fit_transform(data[['CathLabDataCathLabDataInitialTIMI']])
# apply heuristic method for handling miss values in 'CathLabDataCathLabDataFinalTIMI'
data = data.replace({'CathLabDataCathLabDataFinalTIMI':np.nan}, avg)

# convert data types
convert_dict = {'CathLabDataCathLabDataInitialTIMI': 'int64',
                'CathLabDataCathLabDataFinalTIMI':'int64'}
data = data.astype(convert_dict)

# create an empty column for target values
data['target'] = np.nan

# a method for creating target values from differentiate between admission date and discharge date
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

# removing remaining unused columns for creating model
data = data.drop(['AdmissionPainOnsetDate', 'DemographicsDemographicsDateofDischarge', 'OtherDataResultsresult'], axis=1)

# one method to encode categorical values
# data = pd.get_dummies(data, prefix_sep='_')

# one method to encode categorical values
label_encoder = preprocessing.LabelEncoder()
for i in range(0, data.shape[1]):
    if data.dtypes[i]=='object':
        data[data.columns[i]] = label_encoder.fit_transform(data[data.columns[i]])
# creating two classes in target according to problem requirement
data.loc[data['target'] < 6, 'target'] = 1
data.loc[data['target'] >= 6, 'target'] = 2

# move target to last column
data = data[[c for c in data if c not in ['target']] + ['target']]

last_numerical_index = data.columns.get_loc("Age_On_Admission")

def getData():
    return data




