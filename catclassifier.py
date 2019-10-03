# Deep Neural Network for Image Classification: Application
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_training_testing_data(train, test):
    # Loading the data files
    train_file = h5py.File(train, 'r')
    test_file = h5py.File(test, 'r')
    
    # Extracting the arrays from File object
    x_train = train_file['train_set_x'].value
    y_train = train_file['train_set_y'].value
    x_test = test_file['test_set_x'].value
    y_test = test_file['test_set_y'].value

    train_file.close()
    test_file.close()

    plt.imshow(x_train[0])
        
    # reshaping to convert X an Y to 2D array
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    # Normalising the values to floats between 0 to 1
    x_train = x_train/255
    x_test = x_test/255
    
    # Getting the values of m_train, m_test, and n
    m_train = x_train.shape[1]
    m_test = x_test.shape[1]
    n = x_train.shape[0]
    
    # Returning all the data extracted
    return x_train, y_train, x_test, y_test


# BUILDING THE MODEL
# Two-layer neural network
# a 2-layer neural network with the following activation function structure: LINEAR -> RELU -> LINEAR -> SIGMOID

# n_x -- size of the input layer
# n_h -- size of the hidden layer
# n_y -- size of the output layer
n_x = 12288 #(64x64x3) flatten image pixel value
n_h = 7
n_y = 1
layer_dims = (n_x,n_h,n_y)

def two_layer_model(X, Y, layers_dims , learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    print(len(parameters), " len of param")
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        # print(X.shape, "x shape")
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        # print(len(parameters), " len of param")

        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# STEP 1 Load dataset
train_x, train_y, test_x, test_y=  get_training_testing_data(os.getcwd()+'\\input\\train_data.h5', os.getcwd()+'\\input\\test_data.h5')

# show example of picture
index = 10
# ravel back flatten image data from 1 dim to 64x64x3 dim
plt.imshow((train_x[index]).reshape((64,64,3)))
print ("y = " + str(train_y[index]) + ". It's a " + convertLabeltoClass(train_y,index)+  " picture.")
plt.show()


# STEP 2 Initialize Parameters
train_x = train_x.T
test_x = test_x.T
train_y = train_y.T
test_y = test_y.T
paramaters = two_layer_model(train_x,train_y,layers_dims = (n_x,n_h,n_y), num_iterations=2500, print_cost = True)
# print(len(paramaters), "length of param")

# STEP 3    Prediction and test
predictions_train = predict(train_x,train_y,paramaters)
predictions_test = predict(test_x,test_y,paramaters)


print_mislabeled_images(test_x, test_y, predictions_test)