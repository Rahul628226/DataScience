#Given is a dataframe showing name, occupation,salary of people. Find the average salary per occupation.
import pandas as pd
df={
    'Name':['Amal','Abin','Niranjan','Rahul','Sagar'],
    'occ':['Teacher','Business','Engineer','Teacher','Engineer'],
    'salary':[60000,70000,35000,55000,40000]
}
data=pd.DataFrame(df)
print()
avg_salary = data.groupby('occ')['salary'].mean()
print(avg_salary)