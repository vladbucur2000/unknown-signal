import utilities
import numpy as np
from matplotlib import pyplot as plt
import sys

np.random.seed(seed=12)

def squared_error(y_hat, y):
    return np.sum((y_hat - y)**2)

def least_squares_linear(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_order_2(x, y):
    ones = np.ones(x.shape)
    x2 = np.square(x)
    x_e = np.column_stack((ones, x, x2))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_order_3(x, y):
    ones = np.ones(x.shape)
    x2 = np.power(x, 2)
    x3 = np.power(x, 3)
    x_e = np.column_stack((ones, x, x2, x3))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_order_4(x, y):
    ones = np.ones(x.shape)
    x2 = np.square(x)
    x3 = np.power(x, 3)
    x4 = np.power(x, 4)
    x_e = np.column_stack((ones, x, x2, x3, x4))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_order_5(x, y):
    ones = np.ones(x.shape)
    x2 = np.square(x)
    x3 = np.power(x, 3)
    x4 = np.power(x, 4)
    x5 = np.power(x, 5)
    x_e = np.column_stack((ones, x, x2, x3, x4, x5))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_sin(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.sin(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_cos(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.cos(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_cos(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.cos(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v

def least_squares_e(x, y):
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, np.exp(x)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v


def shuffle_input(newarr):
    np.take(newarr,np.random.permutation(newarr.shape[0]),axis=0,out=newarr)
    return newarr

ap1, ap2, ap3, ap4, ap5, apsin, ape, apcos = 0, 0, 0, 0, 0, 0, 0, 0

total_error1, total_error2, total_error3, total_error4, total_error5, total_errorsin, total_errore = 0, 0, 0, 0, 0, 0, 0
ERROR_2, ERROR_3, ERROR_4, ERROR_5, ERROR_SIN, ERROR_COS, ERROR_EXP = 0, 0, 0, 0,0,0,0
smaller2, smaller3 = 0, 0


def kfold(x, y, k):

    global total_error1, total_error2, total_error3, total_error4, total_error5, total_errorsin, total_errore
    folds = np.empty((20, 2))
    
    for t in range(20):
        folds[t] = np.array([x[t], y[t]])

    folds = np.array_split(folds, k)
    
    errors_1, errors_2, errors_3, errors_4, errors_5, errors_sin, errors_cos, errors_e = [], [], [], [], [], [], [], [] 
    
    for i in range(k):
        train = folds.copy() 
        test = folds[i]
        del train[i]
        #print(train)
        newshape = int((20 / k) * (k - 1))
        train = np.stack(train).reshape(newshape, 2)

        #print('train', train)
        #print('test', test)

        ##linear
        a, b = least_squares_linear(train[:,0], train[:,1])
        y_hat_1 = a + b * test[:,0]
                
        ##poly order 2
        a, b, c = least_squares_order_2(train[:,0], train[:,1])
        y_hat_2 = a + b * test[:,0] + c * (test[:,0] ** 2)

        ##poly order 3
        a, b, c, d = least_squares_order_3(train[:,0], train[:,1])
        y_hat_3 = a + b * test[:,0] + c * (test[:,0] ** 2) + d * (test[:,0] ** 3)

        ##poly order 4
        a, b, c, d, e = least_squares_order_4(train[:,0], train[:,1])
        y_hat_4 = a + b * test[:,0] + c * (test[:,0] ** 2) + d * (test[:,0] ** 3) + e * (test[:,0] ** 4)

        ##poly order 5
        a, b, c, d, e, f = least_squares_order_5(train[:,0], train[:,1])
        y_hat_5 = a + b * test[:,0] + c * (test[:,0] ** 2) + d * (test[:,0] ** 3) + e * (test[:,0] ** 4) + f * (test[:,0] ** 5)

        ##sin
        a, b = least_squares_sin(train[:,0], train[:,1])
        y_hat_sin = a + b * np.sin(test[:,0])

        ##cos
        a, b = least_squares_cos(train[:,0], train[:,1])
        y_hat_cos = a + b * np.cos(test[:,0])

        #e^x
        a, b = least_squares_e(train[:,0], train[:,1])
        y_hat_e = a + b * np.exp(test[:,0])

        error_1 = squared_error(y_hat_1, test[:,1])
        error_2 = squared_error(y_hat_2, test[:,1])
        error_3 = squared_error(y_hat_3, test[:,1])
        error_4 = squared_error(y_hat_4, test[:,1])
        error_5 = squared_error(y_hat_5, test[:,1])
        error_sin = squared_error(y_hat_sin, test[:,1])
        error_cos = squared_error(y_hat_cos, test[:,1])
        error_e = squared_error(y_hat_e, test[:,1])
    
       
        errors_1.append(error_1)
        errors_2.append(error_2)
        errors_3.append(error_3)
        errors_4.append(error_4)
        errors_5.append(error_5)
        errors_sin.append(error_sin)
        errors_cos.append(error_cos)
        errors_e.append(error_e)

       
    errors = np.empty(8)
    errors[0] = np.mean(errors_1)
    errors[1] = np.mean(errors_2)
    errors[2] = np.mean(errors_3)
    errors[3] = np.mean(errors_4)
    errors[4] = np.mean(errors_5)
    errors[5] = np.mean(errors_sin)
    errors[6] = np.mean(errors_e)
    errors[7] = np.mean(errors_cos)
   
    total_error1 += errors[0]
    total_error2 += errors[1]
    total_error3 += errors[2]
    total_error4 += errors[3]
    total_error5 += errors[4]
    total_errorsin += errors[5]
    total_errore += errors[6]

    #min_error = min(errors[0],errors[1],errors[2],errors[3],errors[4],errors[6])
    min_error = min(errors)
    global ap1, ap2, ap3, ap4, ap5, apsin, apcos, ape, smaller2, smaller3
    global ERROR_2, ERROR_3, ERROR_4, ERROR_5, ERROR_SIN, ERROR_COS, ERROR_EXP

    if min_error == errors[0]: 
        ap1 += 1
        #print('linear')
    if min_error == errors[1]: 
        ap2 += 1
        ERROR_2 += errors[1]
        ERROR_3 += errors[2]
        ERROR_4 += errors[3]
        ERROR_5 += errors[4]
        smaller2 += errors[2] / errors[1]
        print('here2>3:', errors[2] / errors[1], errors[1], errors[2])
        #print('poly order 2')
    if min_error == errors[2]: 
        ap3 += 1
        ERROR_2 += errors[1]
        ERROR_3 += errors[2]
        ERROR_4 += errors[3]
        ERROR_5 += errors[4]
        smaller3 += errors[1] / errors[2]
        print('here3>2:', errors[1] / errors[2], errors[1], errors[2])

        #print('poly order 3')
    if min_error == errors[3]:
        ap4 += 1
        #print('poly order 4')
    if min_error == errors[4]: 
        ap5 += 1
        #print('poly order 5')
    if min_error == errors[5]: 
        apsin += 1
        ERROR_SIN += errors[5]
        ERROR_COS += errors[7]
        ERROR_EXP += errors[6]
        #print('sinus')
    if min_error == errors[6]:
        ape += 1
        ERROR_SIN += errors[5]
        ERROR_COS += errors[7]
        ERROR_EXP += errors[6]
        #print('exponential')
    if min_error == errors[7]:
        ERROR_SIN += errors[5]
        ERROR_COS += errors[7]
        ERROR_EXP += errors[6]
        apcos += 1


    
def kfold_crossvalidation(Kvalue):
    selected = np.empty((2, 20))
    index = 0
   

    input_files = ['train_data/basic_1.csv', 'train_data/basic_2.csv', 'train_data/basic_3.csv', 'train_data/basic_4.csv', 'train_data/basic_5.csv', 'train_data/adv_1.csv', 'train_data/adv_2.csv', 'train_data/adv_3.csv', 'train_data/noise_1.csv', 'train_data/noise_2.csv', 'train_data/noise_3.csv' ]
    #input_files = ['train_data/basic_3.csv']
    
    for file in input_files:
        points = utilities.load_points_from_file(file)
        #utilities.view_data_segments(points[0], points[1])
      
        for i in range(len(points[0])):
            selected[0][index] = points[0][i]
            selected[1][index] = points[1][i]
            
            if ((i+1) % 20 == 0):
                ##SHUFFLER
                
                newarr = np.empty((20, 2))
                
                for k in range(20):
                   newarr[k] = np.array([selected[0][k], selected[1][k]])

                newarr = shuffle_input(newarr)

                for k in range(20):
                    selected[0][k] = newarr[k][0]
                    selected[1][k] = newarr[k][1]
                
                ##SHUFFLER

                #print(selected)
                kfold(selected[0], selected[1], Kvalue)

                ##CLEAN 20 POINT SET
                selected = np.empty((2, 20))
                index = 0

            else: index = index + 1
    
    print('Linear:', ap1)
    print('Poly ord2:', ap2)
    print('Poly ord3:', ap3)
    print('Poly ord4:', ap4)
    print('Poly ord5:', ap5)
    print('Sin:', apsin)
    print('exponential: ', ape)
    print('cosinus: ', apcos)

    
    print('ERROR2', ERROR_2/(ap2 + ap3 + ap4 + ap5))
    print('ERROR3', ERROR_3/(ap2 + ap3 + ap4 + ap5))
    print('ERROR4', ERROR_4/(ap2 + ap3 + ap4 + ap5))
    print('ERROR5', ERROR_5/(ap2 + ap3 + ap4 + ap5))
    print('smaller2', smaller2)
    print ('smaller3', smaller3)
   # print('ERROR SIN', ERROR_SIN / (apsin + ape + apcos))
   # print('ERROR_COS', ERROR_COS / (apsin + ape  + apcos))
    #print('ERROR_EXP', ERROR_EXP / (ape + apsin + apcos))



def solve(file_name, plot):
    #READ FROM CSV
    points = utilities.load_points_from_file('train_data/' + file_name)
    SPLIT = 16
    if plot == True: 
        ### PRINT POINTS ###
        fig, ax = plt.subplots()
        ax.scatter(points[0], points[1])
    
    selected = np.empty((2, 20))
    index = 0
    total_error = 0

    for i in range(len(points[0])):
        selected[0][index] = points[0][i]
        selected[1][index] = points[1][i]
        if ((i+1) % 20 == 0): 
            ##linear
            a, b = least_squares_linear(selected[0,:SPLIT], selected[1, :SPLIT])
            y_hat_1 = a + b * selected[0]
            
            ##poly order 2
            a, b, c = least_squares_order_2(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_2 = a + b * selected[0] + c * (selected[0] ** 2)

            ##poly order 3
            a, b, c, d = least_squares_order_3(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_3 = a + b * selected[0] + c * (selected[0] ** 2) + d * (selected[0] ** 3)
            ##poly order 4
            a, b, c, d, e = least_squares_order_4(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_4 = a + b * selected[0] + c * (selected[0] ** 2) + d * (selected[0] ** 3) + e * (selected[0] ** 4)

        ##poly order 5
            a, b, c, d, e, f = least_squares_order_5(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_5 = a + b * selected[0] + c * (selected[0] ** 2) + d * (selected[0] ** 3) + e * (selected[0] ** 4) + f * (selected[0] ** 5)


            ##sin
            a, b = least_squares_sin(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_sin = a + b * np.sin(selected[0])

            ##cos
            a, b = least_squares_cos(selected[0, :SPLIT], selected[1, :SPLIT])
            y_hat_cos = a + b * np.cos(selected[0])


            error_1 = squared_error(y_hat_1, selected[1])
            error_3 = squared_error(y_hat_3, selected[1])
            error_sin = squared_error(y_hat_sin, selected[1])
            error_cos = squared_error(y_hat_cos, selected[1])
            total_error_1 = squared_error(y_hat_1, selected[1])
            total_error_3 = squared_error(y_hat_3, selected[1])
            total_error_sin = squared_error(y_hat_sin, selected[1])

            min_error = min([error_1, error_3, error_sin])

            if plot == True: 
                if (error_1 == min_error): 
                    total_error += total_error_1
                    #ax.plot(selected[0], y_hat_1, c = 'g')
                    print('gr1')
                if (error_3 == min_error):
                    total_error += total_error_3
                   # print('error2', squared_error(y_hat_2,selected[1]))
                    ax.plot(selected[0], y_hat_3, c ='r')
                    ax.plot(selected[0], y_hat_2, c = 'y')
                    ax.plot(selected[0], y_hat_4, c = 'b')
                    ax.plot(selected[0], y_hat_5, c = 'pink')
                    plt.legend(["polynomial of order 3", "poylnomial of order 2", "polynomial of order 4", "polynomial of order 5"], loc ="lower right")
                    print('gr3')
                if (error_sin == min_error): 
                    total_error += total_error_sin

                   # ax.plot(selected[0], y_hat_sin, c = 'g')
                    print('gr sin')

            ##CLEAN 20 POINT SET
            selected = np.empty((2, 20))
            index = 0

        else: index = index + 1
    
    print(total_error)
    plt.show()

arguments = sys.argv[1:]

if (len(arguments) == 0): print('Not enough arguments given!')
elif (len(arguments) == 1): solve(arguments[0], False)
elif arguments[1] == '--plot' : solve(arguments[0], True)
elif arguments[1] != '--plot' : print('Optional parameter is wrong!')


#simple_split_crossvalidation()

#K=5
#kfold_crossvalidation(5)

#leave one out
#kfold_crossvalidation(20)