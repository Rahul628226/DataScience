#program to compute sum of all elements, sum of each column and sum of each row.

import numpy as np
arr=([[5,4],[3,2]])
print(np.sum(arr))
print(np.sum(arr,axis=0))
print(np.sum(arr,axis=1))