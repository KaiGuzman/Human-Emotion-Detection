
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import os


# In[12]:


def mmi():
    """
    Load mmi data
    """

    data = pd.read_csv('../Dataset/MMI_OHE.csv')
    train_y = []
    train_x = []
   
    train_X = data.iloc[:, 1:]
    train_Y = data[:][['Emotion']]

    train_x = np.asarray(train_X)
    train_y = np.asarray(train_Y)
    del(train_X)
    del(train_Y)
    
    return (train_x, train_y)


# In[13]:


t1,t2 = mmi()

