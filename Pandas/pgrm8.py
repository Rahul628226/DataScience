#Given a dataframe , select first 2 rows and output them

import pandas as pd
player={
    'Name':['Messi','Mbappe','Neymar'],
    'Goals':[6,7,3]
}
ind=[1,2,3]
data=pd.DataFrame(player,ind)
o = data['Name']
print(o)