import pyreadr
import rpy2.robjects as robjects
from rpy2.robjects import r
import os
import pandas as pd

# # list all rds files in the directory
# files_in_dir = os.listdir('../assignments/assignment1')
# file = 'df_cooling_1.rds'
# read_rds = robjects.r['readRDS']
# r_data = read_rds('../assignments/assignment1/' + file)
#
# if 'data.frame' in r_data.rclass:
#         df = pd.DataFrame({col: list(r_data.rx2(col)) for col in r_data.names})
#         df.head()  # Display the first few rows
#
# for file in files_in_dir:
#     if file.endswith('.rds'):
#         read_rds = robjects.r['readRDS']
#         r_data = read_rds('../assignments/assignment1/' + file)
#
#         if 'data.frame' in r_data.rclass:
#             df = pd.DataFrame({col: list(r_data.rx2(col)) for col in r_data.names})
#             df['date_time'] = pd.to_datetime(df['date_time'], unit='s')
#             df.to_csv('../assignments/assignment1/' + file.replace('.rds', '.csv'), index=False)

# list all rds files in the directory
# files_in_dir = os.listdir('../assignments/assignment2')
#
# for file in files_in_dir:
#     if file.endswith('.rds'):
#         read_rds = robjects.r['readRDS']
#         r_data = read_rds('../assignments/assignment2/' + file)
#
#         if 'data.frame' in r_data.rclass:
#             df = pd.DataFrame({col: list(r_data.rx2(col)) for col in r_data.names})
#             df['date_time'] = pd.to_datetime(df['date_time'], unit='s')
#             df.to_csv('../assignments/assignment2/' + file.replace('.rds', '.csv'), index=False)

files_in_dir = os.listdir('../assignments/assignment4')

for file in files_in_dir:
    if file.endswith('.rds'):
        read_rds = robjects.r['readRDS']
        r_data = read_rds('../assignments/assignment4/' + file)

        if 'data.frame' in r_data.rclass:
            df = pd.DataFrame({col: list(r_data.rx2(col)) for col in r_data.names})
            df['date_time'] = pd.to_datetime(df['date_time'], unit='s')
            df.to_csv('../assignments/assignment4/' + file.replace('.rds', '.csv'), index=False)