
import pandas as pd


filename = '/home/mohsen/Desktop/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
headers = list(data)
s=data['GC1GeneralCharacteristicsSex'].value_counts()
data = data.drop(['Patientid', 'encounterid', 'AdmissionAdmissionProfileNumber',
                  'PrimaryLast','نامبيمار', 'main', 'PMH1PastMedicalHistrySuccessfulCPR',
                  'DemographicDataDemographicSex', 'D41stLesionPCI1stLesionPCIProcedureType',
                  ], axis=1)
print("ha")