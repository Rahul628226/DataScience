#program to create a 4x4 array with random values, now create a new array after swapping the first and last row.

import numpy as np
nums = np.arange(16, dtype='int').reshape(-1, 4)
print("original array:")
print(nums)
print("\nNew array after swapping first and last rows ")
nums[[0,-1],:]=nums[[-1,0],:]
print(nums)