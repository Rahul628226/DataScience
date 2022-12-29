#Write a Python programming to display a pie chart of the popularity of programming
#Languages.

import matplotlib.pyplot as plt
x = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
plt.pie(popularity,labels=x)
plt.show()
