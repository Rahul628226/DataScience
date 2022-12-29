#Write a python program to implement List-to-Series Conversion

import pandas as pd
list = [25,20,15,10,5,1]
print("Original list:")
print(list)
print("Convert the list to a Series:")
res = pd.Series(list)
print(res)