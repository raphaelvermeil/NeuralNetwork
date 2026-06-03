import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


input_layer_size = 784
number_of_hidden_layers = 2
size_of_hidden_layers = [100, 10]
output_layer_size = 10


def init_params():
    array_of_weights_matrices = [np.ndarray] * (number_of_hidden_layers + 1)
    array_of_biases_matrices = [np.ndarray] * (number_of_hidden_layers + 1)
    
    #The first weights matrix because it depends of the input layer size
    array_of_weights_matrices[0] = np.random.rand(size_of_hidden_layers[0], input_layer_size) - 0.5
    array_of_biases_matrices[0] = np.random.rand(size_of_hidden_layers[0], 1) - 0.5

    #The rest of the hidden layers
    for i in range(1, number_of_hidden_layers):
        array_of_weights_matrices[i] = np.random.rand(size_of_hidden_layers[i], size_of_hidden_layers[i - 1]) - 0.5
        array_of_biases_matrices[i] = np.random.rand(size_of_hidden_layers[i], 1) - 0.5


    #The last weight matrix because depends on the output layer size
    array_of_weights_matrices[-1] = np.random.rand(output_layer_size, size_of_hidden_layers[-1]) - 0.5
    array_of_biases_matrices[-1] = np.random.rand(output_layer_size, 1) - 0.5

 
    return array_of_weights_matrices, array_of_biases_matrices

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(array_of_weights_matrices, array_of_biases_matrices, X):
    #Declation
    array_of_Z_matrices = [np.ndarray] * (number_of_hidden_layers + 1)
    array_of_activated_matrices = [np.ndarray] * (number_of_hidden_layers + 1)

    #First Hidden Layer
    array_of_Z_matrices[0] = array_of_weights_matrices[0].dot(X) + array_of_biases_matrices[0]
    array_of_activated_matrices[0] = ReLU(array_of_Z_matrices[0])

    #Hidden Layers
    for i in range(1, number_of_hidden_layers):
        array_of_Z_matrices[i] = array_of_weights_matrices[i].dot(array_of_activated_matrices[i - 1]) + array_of_biases_matrices[i]
        array_of_activated_matrices[i] = ReLU(array_of_Z_matrices[i])


    #Output Layer
    array_of_Z_matrices[-1] = array_of_weights_matrices[-1].dot(array_of_activated_matrices[-2]) + array_of_biases_matrices[-1]
    array_of_activated_matrices[-1] = softmax(array_of_Z_matrices[-1])


    return array_of_Z_matrices, array_of_activated_matrices


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(array_of_weights_matrices, array_of_Z_matrices, array_of_activated_matrices, Y, X):

    #Declarations
    one_hot_Y = one_hot(Y)
    m = Y.size
    array_of_error_Z_matrices = [np.ndarray] * (number_of_hidden_layers + 1)
    array_of_error_weight_matrices = [np.ndarray] * (number_of_hidden_layers + 1)
    array_of_error_bias_matrices = [np.ndarray] * (number_of_hidden_layers + 1)

    #Output Layer
    array_of_error_Z_matrices[-1] = array_of_activated_matrices[-1] - one_hot_Y
    array_of_error_weight_matrices[-1] = (1/m) * array_of_error_Z_matrices[-1].dot(array_of_activated_matrices[-2].T)
    array_of_error_bias_matrices[-1] = (1/m) * np.sum(array_of_error_Z_matrices[-1])

    #Hidden Layers
    for i in range(-2, -(number_of_hidden_layers + 1), -1):
        array_of_error_Z_matrices[i] = array_of_weights_matrices[i + 1].T.dot(array_of_error_Z_matrices[i + 1]) * deriv_ReLU(array_of_Z_matrices[i])
        array_of_error_weight_matrices[i] = (1/m) * array_of_error_Z_matrices[i].dot(array_of_activated_matrices[i - 1].T)
        array_of_error_bias_matrices[i] = (1/m) * np.sum(array_of_error_Z_matrices[i])

    #First Hidden Layer
    array_of_error_Z_matrices[0] = array_of_weights_matrices[1].T.dot(array_of_error_Z_matrices[1]) * deriv_ReLU(array_of_Z_matrices[0])
    array_of_error_weight_matrices[0] = (1/m) * array_of_error_Z_matrices[0].dot(X.T)
    array_of_error_bias_matrices[0] = (1/m) * np.sum(array_of_error_Z_matrices[0])


    return array_of_error_Z_matrices, array_of_error_weight_matrices, array_of_error_bias_matrices


#Subtract the error from the original value
def update_params(array_of_weights_matrices, array_of_biases_matrices, array_of_error_weight_matrices, array_of_error_bias_matrices, alpha):

    for i in range(0, number_of_hidden_layers + 1):
        array_of_weights_matrices[i] = array_of_weights_matrices[i] - alpha * array_of_error_weight_matrices[i]
        array_of_biases_matrices[i] = array_of_biases_matrices[i] - alpha * array_of_error_bias_matrices[i]

    return array_of_weights_matrices, array_of_biases_matrices


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size



""" Takes the following inputs:
X is matrix of the train data -> Each column is a data point. For example if training on images, 1 column is 1 image, # rows for # of pixels
Y is labels of training data -> 1 row with each column being a data point. If images, then # columns is # of pixels
Alpha is learn rate
Iterations is... well iterations, what can I say. Hakuna Matata
"""
def gradient_descent(X, Y, alpha, iterations):
    array_of_weights_matrices, array_of_biases_matrices = init_params()
    for i in range(0, iterations):
        array_of_Z_matrices, array_of_activated_matrices = forward_prop(array_of_weights_matrices, array_of_biases_matrices, X)
        array_of_error_Z_matrices, array_of_error_weight_matrices, array_of_error_bias_matrices = back_prop(array_of_weights_matrices, array_of_Z_matrices, array_of_activated_matrices, Y, X)
        array_of_weights_matrices, array_of_biases_matrices = update_params(array_of_weights_matrices, array_of_biases_matrices, array_of_error_weight_matrices, array_of_error_bias_matrices, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(array_of_activated_matrices[-1])
            print(get_accuracy(predictions, Y))
    return array_of_weights_matrices, array_of_biases_matrices
    


array_of_weights_matrices, array_of_biases_matrices = gradient_descent(X_train, Y_train, 0.10, 500)



def make_predictions(X, array_of_weights_matrices, array_of_biases_matrices):
    array_of_Z_matrices, array_of_activated_matrices = forward_prop(array_of_weights_matrices, array_of_biases_matrices, X)
    predictions = get_predictions(array_of_activated_matrices[-1])
    return predictions

def test_prediction(index, array_of_weights_matrices, array_of_biases_matrices):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], array_of_weights_matrices, array_of_biases_matrices)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


