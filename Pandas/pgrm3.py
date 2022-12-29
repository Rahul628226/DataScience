#Given a dictionary, convert it into a corresponding dataframe and display it.

import pandas as pd
dict = {'name': ['Rahul', 'Abin', 'Amal'],
              'age' : [23, 22, 23],
              'place' : ['Kannur', 'Kasargod', 'Kannur']}
df= pd.DataFrame(dict)
print(df)