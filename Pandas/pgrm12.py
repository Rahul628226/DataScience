# Given are 2 dataframes, with one dataframe containing Employee ID (eid), Employee Name
# (ename) and Stipend (stipend) and the other dataframe containing Employee ID (eid) and
# designation of the employee (designation). Output the Dataframe containing Employee ID
# (eid), Employee Name (ename), Stipend (stipend) and Position (position)
import pandas as pd

d1 = {
    'eid': [1, 2, 3, 4],
    'ename': ['Amal', 'Abin', 'Arjun', 'Niranjan'],
    'stipend': [60000, 70000, 55000, 40000]
}
d2 = {
    'eid': [1, 2, 3, 4],
    'Position': ['employee', 'employee', 'senior_employee', 'intern']
}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
print(df1)
print(df2)
dataframe = pd.merge(df1, df2, how='inner', on='eid')  # required dataframe
print(dataframe)
