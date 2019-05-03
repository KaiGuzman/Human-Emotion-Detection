import scipy
import numpy as np

def PCA_dim_red(train_x, var):
    # Dimensionality Reduction
    m = train_x.shape[0]
    n = train_x.shape[1]

    Mu = np.mean(train_x, axis=0)
    print(Mu.shape)
    train_x = train_x - Mu

    Sigma = (train_x.T).dot(train_x) / (m - 1)
    U, S, V = np.linalg.svd(Sigma)

    # first 253 components maintain 95% variance
    # first 881 components maintain 99% variance
    tr = 0
    k=1
    while tr < var:
        tr = np.sum(S[:k])/np.sum(S)
        k+=1
    print('Using k = '+str(k)+', '+str(tr)+' of the variance was retained')
    V = V[:,:k]
    Train_x = train_x.dot(V)
   # Test_x = test_x.dot(V)
    return Train_x
