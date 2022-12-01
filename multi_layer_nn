import numpy as np
import h5py
import matplotlib as plt
import scipy
from PIL import Image
from scipy import ndimage
from nn_utils import *

#Load data set from h5py files
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#Reshape/flattens the training and test examples for easier processing
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T #"-1" basically means any number of col so that it remains a rectangular matrix
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T    #"T" finds the transpose, which makes the processing later easier

#Standardize color data to range from 0 to 1
train_x = train_x_flatten/255
test_x = test_x_flatten/255

"""
#Defining constants for a 2 layer model
n_x = 12288     #Size of each input example
n_h = 7         #Size of the inner layers
n_y = 1         #Size of output, which is a single value/probability
layers_dims = (n_x, n_h, n_y)
"""

#Defining constants for a 4 layer model
layers_dims = [12288, 20, 7, 5, 1]  #Each element is the size of that layer

def show_image(index):
    """
    Prints the image at the specific index and identifies it to be
    a cat or non-cat image.
    """
    plt.imshow(train_x_orig[index])
    plt.show()

    if train_y[0, index] == 1:
        print("y = 1. " + "It's a cat.")
    else:
        print("y=0. " + "It's not a cat")

def show_dataset_dimensions():
    """
    Prints the dimensions of the data sets
    """
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID

    Arguments:\n
    X -- input data set, shape (n_x, number of examples); every col is an example\n
    Y -- true label vector (1 if cat, 0 if non-cat), shape (1, number of examples)\n
    layers_dims -- dimensions of layers, tuple in the form (n_x, n_h, n_y)\n
    num_iterations -- number of iterations of the optimization loop\n
    learning_rate -- learning rate of the gradient descent update\n
    print_cost -- if set to True, this will print the cost every 100 iterations

    Returns:\n
    parameters -- a dictionary containing W1, W2, b1, b2
    """
    grads = {}          #Will store the partial derivatives
    costs = []          #Keeps track of cost
    m = X.shape[1]      #Number of examples
    (n_x, n_h, n_y) = layers_dims

    #Initialize parameters dictionary
    parameters = initialize_parameters(n_x, n_h, n_y)

    #Get the parameters from the parameters dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #Loop for gradient descent
    for i in range(0, num_iterations):
        #Forward propagation LINEAR+RELU -> LINEAR+SIGMOID
        #Inputs: X, W1, b1, W2, b2
        #Output: A1, cache1, A2, cache2
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

        #Compute the cost, which represents the difference between A2 (calculated output) and Y (true label)
        #Do note that it is not a simple difference, but if A2=Y, cost = 0
        cost = compute_cost(A2, Y)

        #Initialize backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        #Backward propagation, essentially chain rule for partial derivatives
        #Inputs: "dA2, cache2, cache1"
        #Outputs: "dA1, dW2, db2, dA0(not used)"
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        #Store the derivatives in the gradient dictionary
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        #Update parameters, think about linear approximations
        #This is the descent portion of gradient descent
        parameters = update_parameters(parameters, grads, learning_rate)

        #Retrieve W1, b1, W2, b2 from parameters, used in the next iteration to calculate cost
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        #Print the cost of every 100 iterations, if print_cost = True
        if print_cost and i % 100 == 0:
            print("Cost after iterations {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    #Plot the cost, decreasing cost is good
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Learning rate =' + str(learning_rate))
    plt.show()

    #The optimal parameters are returned
    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    
    Arguments:\n
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)\n
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)\n
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1)\n
    learning_rate -- learning rate of the gradient descent update rule\n
    num_iterations -- number of iterations of the optimization loop\n
    print_cost -- if True, it prints the cost every 100 steps\n
    
    Returns:\n
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)   #This seed achieved good results
    costs = []

    #Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    #Loop for gradient descent
    for i in range(0, num_iterations):
        #Forward propagation [LINEAR + RELU] * (L-1) -> LINEAR+SIGMOID
        AL, caches = L_model_forward(X, parameters)

        #Compute cost
        cost = compute_cost(AL, Y)

        #Backward Propagation
        grads = L_model_backward(AL, Y, caches)

        #Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        #Print the cost every 100 steps
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
        
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#For a 2 layer model
"""
parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost=True)
"""

#For a 4 layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

#Prediction accuracy
prediction_train = predict(train_x, train_y, parameters)
prediction_test = predict(test_x, test_y, parameters)

print_mislabeled_images(classes, test_x, test_y, prediction_test)
