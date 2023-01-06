# Other packages
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
# My packages
import mnist_loader
from network import QuadraticCost, CrossEntropyCost, Network
# Debugger
import pdb

# Read the data
train_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# pdb.set_trace()

# x = np.array([1,2,3,4,5])
# x = x.reshape((5,1))
# network.softmax(x)

# Training the Network
net = Network([784, 30, 10], activation_func="sigmoid", cost_func=CrossEntropyCost)
results = net.SGD(train_data, 30, 10, 0.5, evaluation_data=test_data, 
                    monitor_training_cost=True, monitor_training_accuracy=True,
                    monitor_evaluation_cost=True, monitor_evaluation_accuracy=True)
# Unpacking the the results tuple
training_cost, training_accuracy, evaluation_cost, evaluation_accuracy = results

# Print the final costs and accuracy
print("Final Training Cost: {0}".format(training_cost[-1]))
print("Final Training Accuract {0}%".format(training_accuracy[-1]))
print("Final Testing Cost: {0}".format(evaluation_cost[-1]))
print("Final Test Accuracy: {0}%".format(evaluation_accuracy[-1]))

# For ploting/visualization
# Same plot comparison
fig_1, axs_1 = plt.subplots(2, 1)
fig_1.set_size_inches(5, 6)

axs_1[0].plot(training_cost, label="Training Cost")
axs_1[0].plot(evaluation_cost, label="Validation Cost")
axs_1[0].set_title("Costs vs. Epochs")
axs_1[0].set_xlabel("Epoch")
axs_1[0].set_ylabel("Cost")

axs_1[1].plot(training_accuracy, label="Train Accuracy")
axs_1[1].plot(evaluation_accuracy, label="Validation Accuracy")
axs_1[1].set_title("Model Accuracy vs. Epochs")
axs_1[1].set_xlabel("Epoch")
axs_1[1].set_ylabel("Accuracy (%)")

plt.subplots_adjust(wspace=0.6, hspace=0.4)

# Individual plots
fig_2, axs_2 = plt.subplots(2, 2)
fig_2.set_size_inches(8, 6)

axs_2[0, 0].plot(training_cost)
axs_2[0, 0].set_title("Training Cost vs. Epochs")
axs_2[0, 0].set_xlabel("Epoch")
axs_2[0, 0].set_ylabel("Training Cost")

axs_2[0, 1].plot(training_accuracy)
axs_2[0, 1].set_title("Training Accuracy vs. Epochs")
axs_2[0, 1].set_xlabel("Epoch")
axs_2[0, 1].set_ylabel("Training Accuracy (%)")

axs_2[1, 0].plot(evaluation_cost)
axs_2[1, 0].set_title("Validation Cost vs. Epochs")
axs_2[1, 0].set_xlabel("Epoch")
axs_2[1, 0].set_ylabel("Validation Cost")

axs_2[1, 1].plot(evaluation_accuracy)
axs_2[1, 1].set_title("Validation Accuracy vs. Epochs")
axs_2[1, 1].set_xlabel("Epoch")
axs_2[1, 1].set_ylabel("Validation Accuracy (%)")

plt.subplots_adjust(wspace=0.6, hspace=0.4)
plt.show()