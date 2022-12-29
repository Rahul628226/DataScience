#program to check whether two arrays are equal (element wise) or not

import numpy as np
a1=np.array([0.5,6,0.23])
a2=np.array([0.49999999999,6.00000,0.23])
np.set_printoptions(precision=15)
print(a1==a2)
print(np.equal(a1,a2))