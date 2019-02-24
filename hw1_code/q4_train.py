import numpy as np
from q4_features import q4_features

def q4_train(X, Y, lambdaval, mode):

# Trains the regularized least squares regression model using the closed form 
# solution given the training data X, Y.
#
# INPUT:
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the 
#     i-th element is the correct output value for the i-th input example. 
#  lambda: 'float' regularization hyperparameter
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#
# OUTPUT:
#  theta: a numpy.ndarray vector of size [n x 1] and type 'float'
#         containing the learned model parameters.
#

    # insert your code here
    X = q4_features(X, mode)
    X_transpose_X = np.dot(X.T,X)
    X_transpose_Y = np.dot(X.T,Y)
    U_matrix = np.identity(X_transpose_X.shape[0])
    U_matrix[0,0] = 0
    theta = np.linalg.solve(X_transpose_X + lambdaval * U_matrix, X_transpose_Y)

    return theta

# if __name__=="__main__":
#     import scipy.io as spio
#     S = spio.loadmat('autompg.mat', squeeze_me=True)
#     Xtrain = S['trainsetX']
#     Ytrain = S['trainsetY']
#     theta = q4_train(Xtrain, Ytrain, 0.05, 'quadratic')
    
