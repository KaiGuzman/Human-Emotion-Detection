
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import os


# In[31]:


def mmi():
    """
    Load mmi data
    """

    data = pd.read_csv('../Dataset/MMI_OHE.csv', header=None)
    train_y = []
    train_x = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []

    no_of_samples = len(data)
    no_of_train_samples = int( no_of_samples)

    train_X = data.iloc[: no_of_train_samples, 1:]

    train_Y = data[: no_of_train_samples][[0]]

    train_x = np.asarray(train_X)

    train_y = np.asarray(train_Y)


    return (train_x, train_y)

