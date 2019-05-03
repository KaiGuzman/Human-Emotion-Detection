# For FER2013

import sklearn.linear_model
from load_MMI import mmi
import numpy as np
from PCA import PCA_dim_red

from sklearn.model_selection import LeaveOneOut
#from Landmarks import Landmarks



(train_x, train_y)= mmi()


#(train_x, test_x) = Landmarks(train_x, test_x)

#print(train_x, train_y)

#train_x = PCA_dim_red(train_x, 0.99)

train_y = np.ravel(train_y)
loo = LeaveOneOut()




vals_y = []
Lr_pred = []


errork = np.zeros(train_x.size)

counter=0




# Using C as regularization parameter
C_range = [0.001,0.01,0.1,1,10,100]

for i in C_range:
    LR = sklearn.linear_model.LogisticRegression(penalty='l2', C=i, solver='lbfgs', multi_class='multinomial',
                                                 max_iter=10000)

    total_acc = 0
    counter = 0
    for train_idx, val_idx in loo.split(train_x):
        #print (counter)
        X_train, X_val = train_x[train_idx], train_x[val_idx]
        y_train, y_val = train_y[train_idx], train_y[val_idx]

        LR.fit(X_train, y_train)

        Lr_pred = LR.predict(X_val)

        if Lr_pred == y_val:
            total_acc += 1




        counter+=1
    #print(total_acc, train_x)
    accuracy = float(total_acc * 100 /float(len(train_x)))

    print("LR test accuracy for C=" + str(i) + " is : " + str(accuracy))
