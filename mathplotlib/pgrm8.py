import sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.model_selection import train_test_split as tts
from sklearn import tree
a =load_iris()
x =a.data
y= a.target
xTrain,xTest,yTrain,yTest = tts(x,y,test_size=0.20)
dec = dt(criterion = "entropy")
dec.fit(xTrain,yTrain)
tree.plot_tree(dec)
