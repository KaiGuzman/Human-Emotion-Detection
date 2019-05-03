
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import GaussianNB


# In[14]:


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
    no_of_train_samples = int(0.8 * no_of_samples)
    no_of_test_samples = no_of_samples - no_of_train_samples

    train_X = data.iloc[1: no_of_train_samples, 1:]
    test_X = data.iloc[-no_of_test_samples:, 1:]

    train_Y = data[1: no_of_train_samples][[0]]
    test_Y = data[-no_of_test_samples:][[0]]

    train_x = np.asarray(train_X)
    test_x = np.asarray(test_X)
    train_y = np.asarray(train_Y)
    test_y = np.asarray(test_Y)

    return (train_x, train_y), (test_x, test_y)


# In[20]:


gnb = GaussianNB()
loo = LeaveOneOut()
val_store = []
val_pred_Arr = []
(train_x, train_y), (test_x, test_y) = mmi()
for train_index, val_index in loo.split(train_x):
    x_train, x_val = train_x[train_index], train_x[val_index]
    y_train, y_val = train_y[train_index], train_y[val_index]
    val_pred = gnb.fit(x_train,y_train).predict(x_val)
    val_store.append(y_val)
    val_pred_Arr.append(val_pred)
val_store = np.ravel(val_store)
val_pred_Arr = np.ravel(val_pred_Arr)

count1 = 0
#validation accuracy
val_accuracy = np.mean(val_store == val_pred_Arr)
print(val_accuracy)
        

#testing
y_pred = gnb.predict(test_x)
print(y_pred)
#accuracy
count = 0
for i in range(len(test_y)):
    if y_pred[i] == test_y[i]:
        count+=1
    else:
        continue
accuracy = (count/len(test_y)) *100
print(accuracy)

