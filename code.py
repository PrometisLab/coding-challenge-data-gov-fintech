import pandas as pd
import os

print(os.getcwd())

df = pd.read_csv('loan_dataset.csv')

print('\n')
print(df['loan_status'].value_counts())

print('\n')
print(df.describe())
