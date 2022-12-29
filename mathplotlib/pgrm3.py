#Write a Python program to draw a line using given axis values taken from a text file, with
#suitable labels in the x axis, y axis and a title.

import matplotlib.pyplot as plt
temp = []
sales=[]
f = open('temp.txt')
for row in f:
   row = row.split(' ')
   temp.append(int(row[0]))
   sales.append(int(row[1]))

plt.xlabel('temp')
plt.ylabel('sales')
plt.title("Sales Details")
plt.plot(temp,sales)
