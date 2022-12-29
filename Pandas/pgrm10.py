#Given a dataframe with NaN Values, fill the NaN values with 0.

import pandas as pd

data=pd.read_csv("text.csv")
df =data.fillna(0)
print(df)