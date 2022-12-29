#Write a Python programming to display a bar chart of the popularity of programming
#Languages. Use a different color for each bar

import matplotlib.pyplot as plt
x = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
plt.barh(x,popularity,color=['red','green','black','brown','yellow','blue'])
plt.xlabel("Languages")
plt.ylabel("Popularity")
plt.show()
