import numpy as np

def q4_mse(pred_Y, correct_Y):
# This function calculates the Mean Squared Error given two sets of output
# values, one set corresponding to the correct values, the other set
# representing the output values predicted by a regression model
# INPUT:
#  pred_Y: a numpy.ndarray vector of type 'float' containing m predicted values
#  correct_Y: a numpy.ndarray vector of type 'float' containing m correct values
#
# OUTPUT:
#  err: 'float' representing the Mean Squared Error
#

      # insert your code here
      err = (np.square(pred_Y - correct_Y)).mean()
      return err

# if __name__ =="__main__":
#       pred_Y = np.array([2,4,5,7])
#       correct_Y = np.array([2.3,5,1,9])
#       mse = q4_mse(pred_Y, correct_Y)
#       print(mse)