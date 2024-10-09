# read data from data folder
import os
import pandas as pd

# check if __file__ is defined
if '__file__' in locals():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
else:
    data_path = os.path.join(os.path.dirname(__name__), 'data')


data_file = os.path.join(data_path, 'PoliToDataExtraction.csv')
# find parent parent directory
data = pd.read_csv(data_file, sep=';', decimal=',')

if __name__ == '__main__':
    print(data.head())