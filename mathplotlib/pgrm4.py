#Write a Python program to plot two or more lines on same plot with suitable legends of each line
import matplotlib.pyplot as plt
x=[5,10,15,20]
y=[10,20,50,55]
plt.plot(x,y,label="line1")
x1=[20,40,60,80]
y1=[10,15,30,40]
plt.plot(x1,y1,label="line2")
plt.legend()
plt.show()
