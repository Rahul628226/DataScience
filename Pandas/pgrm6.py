#Given a dataframe sort it by multiple columns

import pandas as pd

df = pd.read_csv('data.csv')
print(df)
df_sorted = df.sort_values(by = ['Name', 'mark'])
print(df_sorted)