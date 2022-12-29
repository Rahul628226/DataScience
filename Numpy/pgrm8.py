#program to save a given array to a text file and load it.

import numpy as np
n=([[1,2,3],[4,5,6]])
header="col1 col2 col3"
np.savetxt("ar.txt",n,fmt="%d",header=header)
result=np.loadtxt("ar.txt")
print(result)