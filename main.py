import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data = np.array(data)
# m, n = data.shape
# np.random.shuffle(data)

# data_dev = data[0:1000].T
# Y_dev = data_dev[0]
# X_dev = data_dev[1:n]
# X_dev = X_dev / 255.

# data_train = data[1000:m].T
# Y_train = data_train[0]
# X_train = data_train[1:n]
# X_train = X_train / 255.
# _,m_train = X_train.shape


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

    # W1 = np.random.rand(10, 784) - 0.5
    # b1= np.random.rand(10, 1) - 0.5
    # W2 = np.random.rand(10, 10) - 0.5
    # b2 = np.random.rand(10, 1) - 0.5
    return array_of_weights_matrices, array_of_biases_matrices

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(array_of_weights_matrices, array_of_biases_matrices, X):

    array_of_Z_matrices = [np.ndarray] * (number_of_hidden_layers + 1)

    array_of_activated_matrices = [np.ndarray] * (number_of_hidden_layers + 1)

    array_of_Z_matrices[0] = array_of_weights_matrices[0].dot(X) + array_of_biases_matrices[0]
    array_of_activated_matrices[0] = ReLU(array_of_Z_matrices[0])

    for i in range(1, number_of_hidden_layers):
        array_of_Z_matrices[i] = array_of_weights_matrices[i].dot(array_of_activated_matrices[i - 1]) + array_of_biases_matrices[i]
        array_of_activated_matrices[i] = ReLU(array_of_Z_matrices[i])

    array_of_Z_matrices[-1] = array_of_weights_matrices[-1].dot(array_of_activated_matrices[-2]) + array_of_biases_matrices[-1]
    array_of_activated_matrices[-1] = softmax(array_of_Z_matrices[-1])
 
    # Z1 = W1.dot(X) + b1
    # A1 = ReLU(Z1)
    # Z2 = W2.dot(A1) + b2
    # A2 = ReLU(Z2)
    # Z3 = W3.dot(A2) + b3
    # A3 = softmax(Z3)


    return array_of_Z_matrices, array_of_activated_matrices


# def one_hot(Y):
#     one_hot_Y = np.zeros((Y.size, Y.max() + 1))
#     one_hot_Y[np.arange(Y.size), Y] = 1
#     one_hot_Y = one_hot_Y.T
#     return one_hot_Y

# def deriv_ReLU(Z):
#     return Z > 0

# def back_prop(W1, W2, Z1, Z2, A1, A2, Y, X):
#     one_hot_Y = one_hot(Y)
#     dZ2 = A2 - one_hot_Y
#     dW2 = (1/m) * dZ2.dot(A1.T)
#     dB2 = (1/m) * np.sum(dZ2)
#     dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
#     dW1 = (1/m) * dZ1.dot(X.T)
#     dB1 = (1/m) * np.sum(dZ1)
#     return dZ1, dZ2, dW1, dW2, dB1, dB2

# def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
#     W2 = W2 - alpha * dW2
#     W1 = W1 - alpha * dW1
#     b1 = b1 - alpha * db1
#     b2 = b2 - alpha * db2
#     return W1, W2, b1, b2

# def get_predictions(A2):
#     return np.argmax(A2, 0)

# def get_accuracy(predictions, Y):
#     print(predictions, Y)
#     return np.sum(predictions == Y) / Y.size

# def gradient_descent(X, Y, alpha, iterations):
#     W1, b1, W2, b2 = init_params()
#     for i in range(0, iterations):
#         Z1, Z2, A1, A2 = forward_prop(W1, W2, b1, b2, X)
#         dZ1, dZ2, dW1, dW2, db1, db2 = back_prop(W1, W2, Z1, Z2, A1, A2, Y, X)
#         W1, W2, b1, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
#         if i % 10 == 0:
#             print("Iteration: ", i)
#             predictions = get_predictions(A2)
#             print(get_accuracy(predictions, Y))
#     return W1, b1, W2, b2
    


# W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)



# def make_predictions(X, W1, b1, W2, b2):
#     _, _, _, A2 = forward_prop(W1, W2, b1, b2, X)
#     predictions = get_predictions(A2)
#     return predictions

# def test_prediction(index, W1, b1, W2, b2):
#     current_image = X_train[:, index, None]
#     prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
#     label = Y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)
    
#     current_image = current_image.reshape((28, 28)) * 255
#     plt.gray()
#     plt.imshow(current_image, interpolation='nearest')
#     plt.show()



array_of_weights_matrices, array_of_biases_matrices = init_params()
forward_prop(array_of_weights_matrices, array_of_biases_matrices, X)