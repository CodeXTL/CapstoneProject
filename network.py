# Standard library
import random
import time
# Third-party libraries
import numpy as np
# My libraries
import mnist_loader as loader
# Python Debugger
import pdb


class QuadraticCost(object):

    @staticmethod
    def cost(y_hat, y):
        """
        Returns the quadratic cost of the predicted output "y_hat" and the desired output "y"
        This is for a single example

        Arguments:
            y_hat : scalar, or 1D vector
                the predicted output
            y : scalar, or 1D vector
                the true output

        Returns:
            cost : scalar
                the calculated cost
        """
        cost = 0.5 * (np.linalg.norm(y_hat - y)**2)
        return cost

    @staticmethod
    def delta(z, y_hat, y):
        """
        Returns the error delta from the output layer if quadratic cost is used.
        This delta value is very important for backprop
        
        Arguments:
            z : scalar, or 1D vector
                the pre-activation value in the output layer, z = w*x + b for the last layer
            y_hat : scalar, or 1D vector
                the predicted output
            y : scalar, or 1D vector
                the true output

        Returns:
            delta : scalar, or 1D vector
                the error from the last layer that backprop uses to determine
                how weights and biases are updated to optimize the model
        """
        delta = (y_hat - y) * sigmoid_derivative(z)
        return delta


class CrossEntropyCost(object):

    @staticmethod
    def cost(y_hat, y):
        """
        Returns the cross=entropy cost of the predicted output "y_hat" and the desired output "y"
        This is for a single example. The np.nan_to_num ensures that inexact values are converted
        to the correct value (0.0).

        Arguments:
            y_hat : scalar, or 1D vector
                the predicted output
            y : scalar, or 1D vector
                the true output

        Returns:
            cost : scalar
                the calculated cost
        """
        cost = np.sum(np.nan_to_num(-y*np.log(y_hat) - (1-y)*np.log(1 - y_hat)))
        return cost

    @staticmethod
    def delta(z, y_hat, y):
        """
        Returns the error delta from the output layer if cross-entropy cost is used.
        This delta value is very important for backprop. 'z' is unused, but kept for 
        consistency
        
        Arguments:
            z : scalar, or 1D vector (Unused)
                the pre-activation value in the output layer, z = w*x + b for the last layer
            y_hat : scalar, or 1D vector
                the predicted output
            y : scalar, or 1D vector
                the true output

        Returns:
            delta : scalar, or 1D vector
                the error from the last layer that backprop uses to determine
                how weights and biases are updated to optimize the model
        """
        delta = (y_hat - y)
        return delta


class Network(object):

    def __init__(self, sizes, activation_func="SIGMOID", cost_func=CrossEntropyCost):
        """
        Constructs a neural network of a given size for each layer

        Arguments:
            sizes : list
                a list of the sizes of each layer
        """
        # Setup the layer information
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize weights and biases
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        # Set the functions
        self.activation_func = activation_func
        self.cost_func = cost_func

    def feedforward(self, x):
        """
        Calculates the cost of inputs based on the current weights and biases

        Arguments:
        x -- a matrix of size (784, m), where m is the # of examples

        Returns:
        a -- the final output calculated based on the weights and biases
        """
        a = x
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def backprop(self, x, y):
        """
        Performs the backprop algorithm

        Arguments:
        x -- the input vector, shape of (784, 1)
        y -- the true label vector for input x, shape of (10, 1) 
        """
        dw = [np.zeros(w.shape) for w in self.weights]  # Note: dw.shape == w.shape
        db = [np.zeros(b.shape) for b in self.biases]   # Note: db.shape == b.shape
        
        # (1) Feedforward 
        a = x
        activations = [x]   # Stores activations for layers 0,1,2,...,(L-1), where L is # of layer
                            # activations[0] is the input x, activations[L] is the calculated/predicted output
        zs = []             # Stores pre-activation values for layers 1,2,3,...,(L-1)
                            # The input layer does not have z
        for w, b in zip(self.weights, self.biases):
            # Linear Pass
            z = np.dot(w, a) + b
            zs.append(z)
            # Activation Function
            a = sigmoid(z)
            activations.append(a)

        # (2) Backward Pass for Output Layer, layer (L-1)
        delta = (self.cost_func).delta(zs[-1], activations[-1], y)
        dw[-1] = np.dot(delta, activations[-2].T)
        db[-1] = delta

        # (3) Loop for Backward Pass from (L-2), (L-3),...,1
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            dw[-l] = np.dot(delta, activations[-l-1].T)
            db[-l] = delta

        # Return the gradients for w and b
        return (dw, db)
    
    def update_mini_batch(self, mini_batch, lr, lmbda, m_total):
        """
        Computes gradients and updates the weights and biases

        Arguments:
        mini_batch -- a list of tuples (x, y)
        lr -- the learning rate

        Returns:
        batch_cost_sum -- the sum of the cost of each training example, this is used elsewhere
        """
        # The gradients for weights and biases
        nebla_w = [np.zeros(w.shape) for w in self.weights]
        nebla_b = [np.zeros(b.shape) for b in self.biases]
        # Loops through every training example in mini_batch
        for x, y in mini_batch:
            # Doing backprop
            delta_nebla_w, delta_nebla_b = self.backprop(x, y)
            # Summing all the weights, biases, and costs for each training example
            nebla_w = [nw + dnw for nw, dnw in zip(nebla_w, delta_nebla_w)]
            nebla_b = [nb + dnb for nb, dnb in zip(nebla_b, delta_nebla_b)]
        # Updating the network's weights and biases, stepping by an amount equal to the learning rate (lr)
        # Regularization is used
        self.weights = [(1-lr*(lmbda/m_total))*w - (lr/len(mini_batch))*nw for w, nw in zip(self.weights, nebla_w)]
        self.biases = [b - (lr/len(mini_batch))*nb for b, nb in zip(self.biases, nebla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, lr, 
            lmbda = 0, 
            evaluation_data=None,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False):
        """
        Performs Stochastic Gradient Descent (SGD), which is the basic algorithm for neural network
        optimization. It allows for batches of different sizes

        Arguments:
            training_data : a list of tuples (x, y)
                the entire training dataset, a list of tuples (x, y)
            epochs : int 
                the number of repeats for training, every repeat loops through the entire training dataset
            mini_batch_size : int
                the size of each mini batch
            lr : float 
                learning rate, also known as step size
            test_data (OPTIONAL) : list of tuples (x, y) 
                the entire testing dataset, used if you want to validate the model
        
        Returns:
            training_cost : list[float]
                list of training cost 
            training_accuracy : list[float]
                list of training accuracy
            evaluation_cost : list[float]
                list of evaluation cost
            evaluation_accuracy : list[float]
                list of evaluation accuracy
        """
        training_cost = []
        training_accuracy = []
        evaluation_cost = []
        evaluation_accuracy = []

        total_time_start = time.perf_counter()
        
        # Getting the number of examples for each dataset
        if evaluation_data:
            m_test = len(evaluation_data)
        m_train = len(training_data)
        # Looping through the epochs
        for j in range(epochs):
            epoch_time_start = time.perf_counter()
            # Training
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, m_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr, lmbda, m_train)
            
            # Computing costs and recording them
            if monitor_training_cost:
                cost = self.evaluate_cost(training_data, lmbda)
                training_cost.append(cost)
            if monitor_training_accuracy:
                num_correct = self.evaluate_accuracy(training_data)
                percent_accuracy = num_correct/m_train * 100
                training_accuracy.append(percent_accuracy)
            if monitor_evaluation_cost:
                cost = self.evaluate_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
            if monitor_evaluation_accuracy:
                num_correct = self.evaluate_accuracy(evaluation_data)
                percent_accuracy = num_correct/m_test * 100
                evaluation_accuracy.append(percent_accuracy)
            
            epoch_time_end = time.perf_counter()
            # Messages to track training progress
            if evaluation_data:
                print("Epoch {0} Evaluation Accuracy: {1} / {2}".format(j, round(evaluation_accuracy[-1]/100 * m_test), m_test), 
                        "\tEpoch Elapsed Time: {:.3f}".format(epoch_time_end-epoch_time_start) + " seconds")
            else:
                print("Epoch {0} complete".format(j), 
                        "\tEpoch Elapsed Time: {:.3f}".format(epoch_time_end-epoch_time_start) + " seconds")
        
        total_time_end = time.perf_counter()
        print("Total Elapsed Time: {:.3f} seconds".format(total_time_end-total_time_start))

        return training_cost, training_accuracy, evaluation_cost, evaluation_accuracy

    def evaluate_cost(self, dataset, lmbda):
        """
        Evaluates the cost for an entire dataset

        Argument(s):
            dataset : list(tuple(x_vector, y_vector))
                the dataset with the input and output
            lmbda : float
                the regularization parameter
        
        Returns:
            cost : float
                the computed cost
        """
        cost = 0.0
        for x, y in dataset:
            # Calculate the predicted output
            y_hat = self.feedforward(x)
            # Calculate the cost (based on the cost func) and sum over all examples
            cost += (self.cost_func).cost(y_hat, y)
        cost += 0.5*(lmbda/len(dataset)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)    #This syntax is know as a generator expression, unlike a list comprehension
        return cost

    def evaluate_accuracy(self, dataset):
        """
        Evaluates the model using the weights and biases at the time.
        It computes the number of correct predictions in the dataset "data"

        Argument(s):
        data -- list of tuples with the testing dataset

        Returns:
        num_correct -- the number of correct predictions
        """
        m = len(dataset)
        results = []
        for x, y in dataset:
            # (1) Computing final output
            y_hat = self.feedforward(x)
            # (2) Devectorize the predicted and true labels
            y_hat = np.argmax(y_hat)
            y = np.argmax(y)
            # (3) Calculating and recording the predicted result for one example
            results.append((y_hat, y))
        # (4) Determines the number of correct predictions out of the entire test dataset
        num_correct = sum(int(x == y) for (x, y) in results)
        return num_correct


#### Activation functions
def sigmoid(z):
    """
    The sigmoid function
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    """
    The softmax function
    """
    numerator = np.exp(z)
    denominator = np.sum(np.exp(z))
    return numerator/denominator

#### Cost functions
# def quadratic_cost(y_hat, y):
#     """
#     Calculates the quadratic cost for one example
#     """
#     cost = np.sum(np.power(y - y_hat, 2))/2
#     return cost

# def quadratic_cost_derivative(y_hat, y):
#     """
#     Calculates the derivative of quadratic cost, with respect to the output prediction
#     """
#     d_cost = np.subtract(y_hat, y)
#     return d_cost

# def cross_entropy_cost(y_hat, y):
#     """
#     Calculates the cross-entropy cost for one example
#     """
#     cost = -np.sum(np.add(np.multiply(y, np.log(y_hat)), np.multiply(1-y, np.log(1-y_hat))))
#     return cost

# def cross_entropy_cost_derivative(y_hat, y):
#     """
#     Calculates the derivative of cross-entropy cost, with respect to the output prediction
#     """
#     d_cost = np.add(np.divide(-y, y_hat), np.divide(1-y, 1-y_hat))
#     return d_cost

def log_likelihood_cost(y_hat, y):
    """
    Calculates the log likelihood cost
    """
    filtered_output = np.sum(np.multiply(y_hat, y)) # filter out the output of interest
    cost = -np.log(filtered_output)
    return cost

#### Regularization functions


#### Not working function
    # def split_into_mini_batches(self, dataset, mini_batch_size):
    #     """
    #     Splits a data set into mini_batches each with a specific number of examples

    #     Arguments:
    #     x -- the input dataset that you wish to split
    #     mini_batch_size -- the size of each mini batch
    #     """
    #     mini_batches = []
    #     # Get the size of x, specifically the total number of examples in x
    #     m = len(dataset)
    #     # Loop to split the dataset
    #     for k in range(0, m, mini_batch_size):
    #         mini_batch = dataset[k: k+mini_batch_size]  # This is a list of tuples, such as one like this (x, y)
    #         x = [mini[0] for mini in mini_batch]        # This is a list of vectors, shape (784, 1)
    #         y = [mini[1] for mini in mini_batch]        # This is a list of vectors, shape (10, 1)
    #         x = np.concatenate(x, axis=1)               # This turns x into a matrix with all the examples, shape (784, n), where n is the mini_batch size
    #         y = np.concatenate(y, axis=1)               # This turns y into a matrix with all the labels, shape (10, n)
    #         mini_batches.append((x, y))
    #     return mini_batches