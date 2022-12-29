#Given a dataframe with custom indexing , convert it to default indexing and display it.

import pandas as pd
player={
    'Name':['Messi','Mbappe','Neymar'],
    'Goals':[6,7,3]
}
ind=[1,2,3]
data=pd.DataFrame(player,ind)
print(data)
dataframe = data.reset_index()
print(dataframe)