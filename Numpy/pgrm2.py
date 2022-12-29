#Program to create an element-wise comparison (greater, greater_equal, lesseer).

import numpy as np
x=([5,10])
y=([5,15])
print('array1=',x)
print('array2=',y)
print(np.greater(x,y))
print(np.greater_equal(x,y))
print(np.equal(x,y))
print(np.less(x,y))
print(np.less_equal(x,y))
