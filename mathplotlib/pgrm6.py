#Consider the following data. Programming languages: Java Python PHP JavaScript C# C++ Popularity 22.2 17.6 8.8 8 7.7 6.7

#Write Python programming to display a bar chart of the popularity of programming
#Languages

import matplotlib.pyplot as plt
x = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
plt.bar(x,popularity,color='green',edgecolor='blue')
plt.xlabel("Languages")
plt.ylabel("Popularity")
plt.show()
