
# coding: utf-8

# In[2]:


import sklearn .svm
import numpy as np
from load_MMI import mmi
from PCA import PCA_dim_red
import numpy as np
import scipy
import os
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV


# In[26]:


#Importing Data
print("Importing data")
(train_x, train_y) = mmi()
print("Imported data, initiating training")
train_x = train_x[1:]
train_y = train_y[1:]

loo = LeaveOneOut()

vals_y = []
Poly_preds_y = []
Gaussian_preds_y = []

SVM1 = sklearn.svm.SVC(C=5.0, kernel='poly', coef0=1.0)
SVM2 = sklearn.svm.SVC(C=20.0, kernel='rbf')


for train_idx, val_idx in loo.split(train_x):
    X_train, X_val = train_x[train_idx], train_x[val_idx]
    y_train, y_val = train_y[train_idx], train_y[val_idx]
    
    SVM1.fit(X_train, y_train)
    SVM2.fit(X_train, np.ravel(y_train))
    Poly_pred_y = SVM1.predict(X_val)
    Gaussian_pred_y = SVM2.predict(X_val)    
        
    vals_y.append(list(y_val))
    Poly_preds_y.append(list(Poly_pred_y))
    Gaussian_preds_y.append(list(Gaussian_pred_y))

vals_y = np.ravel(vals_y)
Poly_preds_y = np.ravel(Poly_preds_y)
Gaussian_preds_y = np.ravel(Gaussian_preds_y)

print('---------------------------- SVM Polynomial----------------------------------------------')
p_accuracy = np.mean(Poly_preds_y == vals_y)
print("Accuracy : " + str(p_accuracy))

# preds_ty = np.ravel(SVM1.predict(test_x))
# test_accuracy = np.mean(preds_ty == test_y)
# print("SVM test accuracy : " + str(test_accuracy))

print('---------------------------- SVM Gaussian-------------------------------------------------')
g_accuracy = np.mean(Gaussian_preds_y == vals_y)
print("Accuracy : " + str(g_accuracy))

# preds_ty = np.ravel(SVM2.predict(test_x))
# test_accuracy = np.mean(preds_ty == test_y)
# print("SVM test accuracy : " + str(test_accuracy))

########### PCA ###################################################################
train_x= PCA_dim_red(train_x, 0.99)
loo = LeaveOneOut()

vals_y = []
Poly_preds_y = []
Gaussian_preds_y = []

SVM1 = sklearn.svm.SVC(C=5.0, kernel='poly', coef0=1.0)
SVM2 = sklearn.svm.SVC(C=5.0, kernel='rbf')


for train_idx, val_idx in loo.split(train_x):
    X_train, X_val = train_x[train_idx], train_x[val_idx]
    y_train, y_val = train_y[train_idx], train_y[val_idx]
    
    SVM1.fit(X_train, y_train)
    SVM2.fit(X_train, np.ravel(y_train))
    Poly_pred_y = SVM1.predict(X_val)
    Gaussian_pred_y = SVM2.predict(X_val)    
        
    vals_y.append(list(y_val))
    Poly_preds_y.append(list(Poly_pred_y))
    Gaussian_preds_y.append(list(Gaussian_pred_y))

vals_y = np.ravel(vals_y)
Poly_preds_y = np.ravel(Poly_preds_y)
Gaussian_preds_y = np.ravel(Gaussian_preds_y)

print('------------------------- SVM Polynomial with PCA----------------------------------------------')
pca_p_accuracy = np.mean(Poly_preds_y == vals_y)
print("Accuracy : " + str(pca_p_accuracy))

# preds_ty = np.ravel(SVM1.predict(test_x))
# test_accuracy = np.mean(preds_ty == test_y)
# print("SVM test accuracy : " + str(test_accuracy))

print('--------------------------SVM Gaussian with PCA-------------------------------------------------')
pca_g_accuracy = np.mean(Gaussian_preds_y == vals_y)
print("Accuracy : " + str(pca_g_accuracy))

# preds_ty = np.ravel(SVM2.predict(test_x))
# test_accuracy = np.mean(preds_ty == test_y)
# print("SVM test accuracy : " + str(test_accuracy))

