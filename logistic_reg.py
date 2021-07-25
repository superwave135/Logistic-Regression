import math
import statistics
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt 
import scipy
from scipy.special import expit
import matplotlib.pyplot as plt
import csv

# pandas dataframe to preview the data
df_raw = pd.read_csv('data_banknote_authentication.txt', names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Class(label)']) # for checking purpose
df_raw.head() # pandas form

# A
def loadData(filename): # load the file into numpy form
    return  pd.read_csv(filename, header = None).to_numpy()

np_raw = loadData('data_banknote_authentication.txt')
print(np_raw, np_raw.shape, type(np_raw))

# B
def dataNorm(X): # normalize the data
    col = np.ones(X.shape[0])
    row = col.reshape(1,X.shape[0])   # reshape ones into [1, 1, ..., 1]  
    for i in X.T:  #loop over X.T to normalize the cols in X
        arr_transpose = (i - np.min(i)) / np.ptp(i)
        row = np.vstack((row, arr_transpose))
    return row.T # transpose row back to original form as X
## ---- calling main ----
X_norm = dataNorm(np_raw)
print(X_norm, X_norm.shape) # check the row and col

# E
def errCompute(X_norm, theta): # compute the error
    # variable initialization
    x = X_norm[:,:-1]
    y = X_norm[:,-1]
    M = X_norm.shape[0]
    y_hat = expit(np.dot(x, theta))
    
    result = (np.dot(y,np.log(y_hat)) + np.dot((1 - y),np.log(1 - y_hat))) / (-M)  # error function
    return result

error_compute = errCompute(X_norm, np.zeros((X_norm.shape[1]-1,1)))
print(f'\nThe compute error J is: {error_compute}\n')

# F
def stochasticGD(X_norm, theta, alpha, num_iters):   
    # variable initialization
    x = X_norm[:,:-1]
    y = np.reshape(X_norm[:,-1],(x.shape[0],1))
    # creating an array to record the error after each iteration
    errRecords = np.zeros((num_iters,1))
    
    # stochasticGD algorithm
    for idx in range(num_iters):
        #print "theta",theta.shape
        i = idx % x.shape[0]
        y_hat = expit(np.dot(x,theta))
        for j in range(x.shape[1]):            
            theta[j] += alpha * (y[i] - y_hat[i]) * x[i][j]
        errRecords[idx] = errCompute(X_norm, theta)
        
    # errCompute() should return 0.3151
    print("errCompute() = ", errCompute(X_norm, theta))
    # accuracy verification
    y_hat = np.around(expit(np.dot(x,theta)))
    accuracy = (y == y_hat).mean() * 100
    print("accuracy = ", accuracy, "%")
    
    # plot of error function against iteration number
    x_axis = [x for x in range(0,num_iters)]
    plt.plot(x_axis,list(errRecords))
    plt.ylabel('Error')
    plt.xlabel('Number of Iterations')
    plt.show()
    
    return theta

# F ----------------- calling main block -------------------
shuffled_raw = loadData('shuffled.data')
print(shuffled_raw, shuffled_raw.shape)
X_shufnorm = dataNorm(shuffled_raw)
print(X_shufnorm, X_shufnorm.shape)
theta = stochasticGD(X_shufnorm, np.zeros((X_shufnorm.shape[1]-1,1)), 0.01, 1372*20)
print('The learned theta is:\n', theta)

# F 
def Predict(X_norm, theta):
    # variable initialization
    x = X_norm[:,:-1]
    print("x",x.shape)
    
    # accuracy verification
    pred_from_data = loadData('predict.data')
    y_hat = np.around(expit(np.dot(x,theta)))
    accuracy_rate = (pred_from_data == y_hat).mean() * 100
    print("accuracy = ", accuracy_rate, "%") 
    return y_hat

## main block to compute accuracy and y_hat on predict.data   
y_hat = Predict(X_shufnorm, theta)
print("The predicted y:\n", y_hat, "%") 

# F
def optimal_alpha(X_norm, theta, alpha, num_iters, cost=0.05):
    # variable initialization
    x = X_norm[:,:-1]
    y = np.reshape(X_norm[:,-1],(x.shape[0],1))
    # creating an array to record the error after each iteration
    err_records = np.zeros((num_iters,1))
    alpha_records = np.zeros((num_iters, 1))
    # stochasticGD algorithm
    for idx in range(num_iters):
        #print "theta",theta.shape
        i = idx % x.shape[0]
        y_hat = expit(np.dot(x,theta))
        for j in range(x.shape[1]):
            prev = theta
            theta[j] += alpha * (y[i] - y_hat[i]) * x[i][j]
            new = theta                    
        err_records[idx] = errCompute(X_norm, theta)
        alpha_records[idx] = alpha      
        if err_records[0] < cost:   
            break
    return err_records[0][0], alpha_records[idx][0]

optimal_theta = np.array([4.40351322, -7.15911589, -2.66363364, -1.3358347, 1.39339848])        

## ---------------------------- calling main block -------------------------------
# given cost = 0.05, iteration = 1372*20, compute the optimal learn rate
error_list = []
for i in np.arange(0.01, 1.0, 0.01): # loop over the learning rate from 0.1 to 1.0
    cost = 0.05                      # abitrary cutoff cost function value 
    err, optimal = optimal_alpha(X_shufnorm, optimal_theta, i, 1372*20, cost)
    error_list.append(err)
    print('At learn rate =', round(i, 2))
    print(f'Cost: {round(err, 3)}, Learn rate: {round(optimal, 5)}')
    print()
    if err < cost:
        print(f'At cutoff_cost of {cost}, the optimal learn rate is {round(optimal, 3)}')
        break

# G
def splitTT(raw, PercentTrain):    
    np.random.shuffle(raw) # shuffles the rows in the X_norm matrix
    row_num = raw.shape[0] # get the num of rows
    ratio = int(row_num*PercentTrain) # ratio expresses in int num
    test = raw[ratio:,:]
    train =  raw[:ratio,:]
#     X_split = [X_train, X_test] # return a list of X train and X test sets
    return train, test

## ---------------- calling main block to split TT------------------
train_set = []
test_set = []
for i in range(5):   
    train, test = splitTT(np_raw, 0.6)
    print(f'train shape: {train.shape}')
    print(f'test shape: {test.shape}')
    print()

    train_norm = dataNorm(train)  # normalize the train
    test_norm = dataNorm(test)
    
    train_set.append(train_norm)
    test_set.append(test_norm)
    
# print(f'shape of train_set and test_set: {train_norm.shape}, {test_norm.shape}')
# print(f'len of train_set and test_set: {len(train_set)}, {len(test_set)}\n')

print(train_set[0])
print(test_set[0])
y = test_set[0][:, -1]
y.shape

for i in range(5): # output all train and test sets to csv files
    with open(f'train_set_{i+1}.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(val for val in train_set[i])
    with open(f'test_set_{i+1}.csv', 'w') as f: 
        write = csv.writer(f) 
        write.writerows(val for val in test_set[i])

# H 
def pred_yhat_part_h(xy_test, theta):

    # variable initialization
    x = xy_test[:,:-1]
#     print("x",x.shape)
    y = xy_test[:, -1]
    y = np.reshape(xy_test[:,-1],(xy_test.shape[0],1))
#     print('y shape:', y.shape)
    
    # accuracy verification
    y_hat = np.around(expit(np.dot(x,theta)))
#     print('y_hat shape:', y_hat.shape)   
    
    accuracy = (y == y_hat).mean() * 100
#     print("accuracy = ", accuracy, "%") 
    return accuracy, y_hat

# H
## -------------------- calling main block ----------------------
theta_temp = []
yhat_temp = []
accuracy_temp = []
for i in range(5):
    print(f'--------- Set {i+1} ---------')
    # given a train set and optimal learn_rate 0.10 
    # call the SGD function to output theta
    theta = stochasticGD(train_set[i], np.zeros((train_set[i].shape[1] - 1, 1)), 0.1, (train_set[0].shape[0])*20)
    
    # call the tweaked Predict function to o/p accuracy and y_hat
    pred_accuracy, y_hat = pred_yhat_part_h(test_set[i], theta)
    print(f'pred_accuracy: {pred_accuracy} %')   
    print('theta:\n', theta)
    theta_temp.append(theta)
    yhat_temp.append(y_hat)
    accuracy_temp.append(pred_accuracy)   
    print()

# append theta to list
theta_list = [] 
for i in theta_temp:
    theta_subset = []
    for j in i:
        theta_subset.append(j[0])
    theta_list.append(theta_subset)

# output theta list to csv file
with open('theta_list.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in theta_list)
    
# append the y_hat to list
yhat_list = [] 
for i in yhat_temp:
    yhat_subset = []
    for j in i:
#         print(j[0])
        yhat_subset.append(j[0])
    yhat_list.append(yhat_subset)

# output y_predict list to csv file
with open('y_hat.csv', 'w', newline='') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in yhat_list)
    
# output accuracy list to csv file
np.savetxt('accuracy.csv', accuracy_temp, fmt='%10.5f', delimiter=',')

#create a table for accuracy list
accur_df = pd.DataFrame(accuracy_temp, columns = ['Accuracy'])
accur_df # output the table form of accuracies

## calling block to output theta list into csv file
theta_list = [] 
for i in theta_temp:
    theta_subset = []
    for j in i:
#         print(j[0])
        theta_subset.append(j[0])
    theta_list.append(theta_subset)

with open('theta_list.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in theta_list)

## calling block to output yhat list into csv file
yhat_list = [] 
for i in yhat_temp:
    yhat_subset = []
    for j in i:
#         print(j[0])
        yhat_subset.append(j[0])
    yhat_list.append(yhat_subset)

with open('y_hat.csv', 'w', newline='') as f: 
    write = csv.writer(f) 
    write.writerows(val for val in yhat_list)

## TEST UNIT to make sure the csv files load okay
yo = pd.read_csv('test_set_1.csv', header=None)
yo

#TEST UNIT to make sure the csv files load okay
hat = pd.read_csv('y_hat.csv', header=None)
hat