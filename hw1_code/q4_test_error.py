import numpy as np
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_test_error(X, Y, Xtest, Ytest, lambdavec, mode):
# Given training and test set, it trains the model and calculates the test error.
#
# INPUT
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input training example
#  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the 
#     i-th element is the correct output value for the i-th input training example. 
#  Xtest: a numpy.ndarray vector of size [M x d] and type 'float', where 
#         each row is a d-dimensional test example
#  Ytest: a numpy.ndarray vector of size [M x 1] and type 'float',
#         containing the output values of the test examples
#  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
#             containing the set of regularization hyperparameter values
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#
# OUTPUT
#  error: a numpy.ndarray vector of size [k x 1] and type 'float'
#         containing the test errors, one for each value in lambdavec.
#

    # insert your code here
    mse = []
    for lamdaval in lambdavec:
        theta = q4_train(X, Y, lamdaval, mode)
        pred_Y = q4_predict(theta, Xtest, mode)  #predict
        error = q4_mse(pred_Y, Ytest)  #find mean square error
        mse.append(error)
    error = np.array(mse)
    return error

if __name__ == '__main__':
    import scipy.io as spio
    import math
    S = spio.loadmat('autompg.mat', squeeze_me=True)
    Xtrain = S['trainsetX']
    Ytrain = S['trainsetY']

    Xtest = S['testsetX']
    Ytest = S['testsetY']
    mode = 'linear'
    lambdavec = np.array([math.pow(10,-5),
                math.pow(10,-3), math.pow(10,-1),
                math.pow(10,1),math.pow(10,3),
                math.pow(10,5), math.pow(10,7)])

    error = q4_test_error(Xtrain, Ytrain, Xtest, Ytest, lambdavec, mode)
    print(error)
    
