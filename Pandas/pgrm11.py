#Given is a dataframe showing Company Names (cname) and corresponding Profits (profit).
#Convert the values of the Profit column such that values in it greater than 0 are set to True
#and the rest are set to False.

import pandas as pd
df={
    'cname':['aa & bb','a12','ccc12','KP Store','D3 & V3'],
    'profit':[60000,70000,-4000,-5000,40000]
}
data= pd.DataFrame(df)
print(data)
print()
data['profit'] = data['profit'].apply(lambda x:x>0)
print(data)
