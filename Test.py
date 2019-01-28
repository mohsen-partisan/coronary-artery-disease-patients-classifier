
import pandas as pd


filename = '/home/mohsen/Desktop/primaryPCI.csv'
data = pd.read_csv(filename, sep=';')
data = data[data.columns[0:64]]
headers = list(data)
data = data.drop(['Patientid'], axis=1)
print("ha")