import pandas as pd
import os

print(os.getcwd())

df = pd.read_csv('loan_dataset.csv',index_col=0,delimiter=";")

"""print('\n')
print(df['loan_status'].value_counts())

print('\n')
print(df.describe())"""

print(df)
