#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_iris


# In[3]:


x,y=load_iris(return_X_y=True)


# In[4]:


x=x[:,2]


# In[5]:


x[:5]


# In[6]:


y[:5]


# In[7]:


xTrain,xTest,yTrain,yTest=tts(x,y,test_size=.20,random_state=2)


# In[8]:


xTrain[:5]


# In[9]:


xTest[:5]


# In[10]:


yTrain[:5]


# In[11]:


yTest[:5]


# In[12]:


xTrain = np.array(xTrain).reshape(-1,1)


# In[13]:


yTrain = np.array(yTrain).reshape(-1,1)


# In[14]:


xTest = np.array(xTest).reshape(-1,1)


# In[15]:


cl=lr()


# In[16]:


cl.fit(xTrain,yTrain)


# In[17]:


yPred = cl.predict(xTest)


# In[18]:


yPred[:5]


# In[20]:


r2_score(yTest,yPred)


# In[21]:


mean_squared_error(yTest,yPred)


# In[24]:


cl.coef_


# In[26]:


plt.scatter(xTest,yTest,color='r')
plt.plot(xTest,yPred,color='b')


# In[ ]:




