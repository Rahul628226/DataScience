#Given a 2D list convert it into a corresponding datframe and display it.

import pandas as pd
l1= [[1, 'Rahul', 23],
         [2, 'Abin', 22],
         [3, 'Amal', 23]]
df = pd.DataFrame(l1, columns = ['id', 'name', 'age'])
print(df)