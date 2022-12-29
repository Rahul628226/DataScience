#Given a CSV file read it into a dataframe and display it

import pandas as pd

dataframe = pd.read_csv('data.csv')
print(dataframe)