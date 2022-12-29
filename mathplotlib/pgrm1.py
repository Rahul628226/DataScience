#Draw a line in a diagram from position (1, 3) to (2, 10) then to (6, 12) and finally to
#position (18, 20). (Mark each point with a beautiful green colour and set line colour to red
#and line style dotted).
import matplotlib.pyplot as plt
import numpy as np

x= np.array([1,2,6,18])
y=np.array([3,10,12,20])
plt.plot(x,y)
plt.show()
