import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

# SSE
def squared_error(y_hat, y):
    return np.sum((y_hat - y)**2)

def least_squares_linear(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_order_3(x, y):
    ones = np.ones(x.shape)
    x2 = np.power(x, 2)
    x3 = np.power(x, 3)
    x_e = np.column_stack((ones, x, x2, x3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_sin(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

fig, ax = plt.subplots()

def kfold(x, y, k):
    folds = np.empty((20, 2))
    
    for t in range(20):
        folds[t] = np.array([x[t], y[t]])

    folds = np.array_split(folds, k)

    errors_1, errors_3, errors_sin = [], [], [] # stores all the CV errors and the model associated to it
    errors1, errors3, errorssin = [], [], [] # stores all the CV erorrs to get the mean after the kfold

    for i in range(k):
        train = folds.copy() 
        test = folds[i]
        del train[i]
        newshape = int((20 / k) * (k - 1))
        train = np.stack(train).reshape(newshape, 2)

        ##linear
        a1 = least_squares_linear(train[:,0], train[:,1])
        y_hat_1 = a1[0] + a1[1] * test[:,0]

        ##poly order 3
        a3 = least_squares_order_3(train[:,0], train[:,1])
        y_hat_3 = a3[0] + a3[1] * test[:,0] + a3[2] * (test[:,0] ** 2) + a3[3] * (test[:,0] ** 3)

        ##sin
        asin = least_squares_sin(train[:,0], train[:,1])
        y_hat_sin = asin[0] + asin[1] * np.sin(test[:,0])

        error_1 = squared_error(y_hat_1, test[:,1])
        error_3 = squared_error(y_hat_3, test[:,1])
        error_sin = squared_error(y_hat_sin, test[:,1])

        errors1.append(error_1)
        errors3.append(error_3)
        errorssin.append(error_sin)

        errors_1.append([error_1, a1])
        errors_3.append([error_3, a3])
        errors_sin.append([error_sin, asin])

    mean_linear = np.mean(errors1)
    mean_poly = np.mean(errors3)
    mean_sin = np.mean(errorssin)

    #used for decision for linear, polynomial or sin
    min_mean = min(mean_linear, mean_poly, mean_sin)

    Min = 9999999999
    
    if min_mean == mean_linear:
        for i in range (k):
            if (errors_1[i][0] < Min): minimum = errors_1[i]

        a1 = minimum[1]
        
        y_hat_1 = a1[0] + a1[1] * x
        ax.plot(x, y_hat_1, c = 'r')
        
        #print('linear')

        return squared_error(y_hat_1, y) #reconstruction error

    if min_mean == mean_poly:
        for i in range (k):
            if (errors_3[i][0] < Min): minimum = errors_3[i]

        a3 = minimum[1]
    
        y_hat_3 = a3[0] + a3[1] * x + a3[2] * (x ** 2) + a3[3] * (x ** 3)
        ax.plot(x, y_hat_3, c = 'r')
        
        #print('poly gr 3')
        
        return squared_error(y_hat_3, y) #reconstruction error

    if min_mean == mean_sin:
        for i in range (k):
            if (errors_sin[i][0] < Min): minimum = errors_sin[i]

        asin = minimum[1]
        
        y_hat_sin = asin[0] + asin[1] * np.sin(x)
        ax.plot(x, y_hat_sin, c = 'r')
        
        #print('sinus')
        
        return squared_error(y_hat_sin, y) #reconstruction error

    

def solve(file_name, plot):
    #READ FROM CSV
    points = load_points_from_file(file_name)

    if plot == True: 
        ### PRINT POINTS ###
        ax.scatter(points[0], points[1])
    
    selected = np.empty((2, 20))
    index = 0
    total_error = 0

    for i in range(len(points[0])):
        selected[0][index] = points[0][i]
        selected[1][index] = points[1][i]
        if ((i+1) % 20 == 0): 
            total_error += kfold(selected[0], selected[1], 20)

            ##CLEAN 20 POINT SET
            selected = np.empty((2, 20))
            index = 0

        else: index += 1

    
    print(total_error)
    if plot == True: plt.show()


arguments = sys.argv[1:]

if (len(arguments) == 0): print('Not enough arguments given!')
elif (len(arguments) == 1): solve(arguments[0], False)
elif arguments[1] == '--plot' : solve(arguments[0], True)
elif arguments[1] != '--plot' : print('Optional parameter is wrong!')
