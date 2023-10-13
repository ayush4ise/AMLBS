import numpy as np

class NN():
    def __init__(self, input_layer_size=2, hidden_layer_size=2, output_layer_size=2, hidden_layers=1, lr=0.1, epochs=1000):
        """
        Initialize a neural network with specified parameters.

        Args:
            input_layer_size (int): Number of input neurons.
            hidden_layer_size (int): Number of neurons in each hidden layer.
            output_layer_size (int): Number of output neurons.
            hidden_layers (int): Number of hidden layers.
            lr (float): Learning rate for weight updates.
            epochs (int): Number of training epochs.
        """
        # Initialize network parameters
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.number_of_epochs = epochs

        # Initialize weights randomly
        self.W = [np.random.randn(self.input_layer_size, self.hidden_layer_size)]
        for layer in range(hidden_layers-1):
            self.W.append(np.random.randn(self.hidden_layer_size, self.hidden_layer_size))
        self.W.append(np.random.randn(self.hidden_layer_size, self.output_layer_size))

    def activation(self, x):
        """
        Sigmoid activation function.

        Args:
            x (float): Input value.

        Returns:
            float: Activated value.
        """
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X, less_steps=False):
        """
        Perform forward propagation.

        Args:
            X (numpy.ndarray): Input data with shape (num_samples, num_features).
            less_steps (bool, optional): If True, perform partial forward propagation.

        Returns:
            numpy.ndarray: Output of the network with shape (num_samples, output_layer_size).
        """
        num_samples = X.shape[0]
        input_list = X

        if not less_steps:
            less_steps = self.hidden_layers + 1 

        outputs = np.zeros((num_samples, self.output_layer_size))

        for step in range(less_steps):
            output_list = np.dot(input_list, self.W[step])
            output_list = [self.activation(neuron) for neuron in output_list]
            input_list = output_list
            outputs = output_list

        return outputs

    
    def backward(self, X, y):
        """
        Perform backward propagation.

        Args:
            X (numpy.ndarray): Input data with shape (num_samples, num_features).
            y (numpy.ndarray): Target output with shape (num_samples, output_layer_size).

        Returns:
            None
        """
        num_samples = X.shape[0]

        for sample in range(num_samples):
            layer_error = self.forward(X[sample]) * (np.array(1) - self.forward(X[sample])) * (y[sample] - self.forward(X[sample]))
            self.W[-1] += self.lr * (np.dot(np.matrix(self.forward(X[sample])).transpose(), layer_error).transpose())

            for step in range(2, self.hidden_layers + 2):
                layer_error = np.array(self.forward(X[sample], self.hidden_layers+2-step)) * (np.array(1) - np.array(self.forward(X[sample], self.hidden_layers+2-step))) * np.dot(self.W[-step+1], np.matrix(layer_error).transpose())
                self.W[-step] += self.lr * (np.dot(np.matrix(self.forward(X[sample], self.hidden_layers+1-step)).transpose(), layer_error).transpose())

    def train(self, X, y):
        """
        Train the neural network.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target output.

        Returns:
            None
        """
        for _ in range(self.number_of_epochs):
            self.backward(X, y)
    
    def total_error(self, X, y):
        """
        Calculate the total error of the network.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target output.

        Returns:
            float: Total error.
        """
        return np.sum((y - self.forward(X))**2 / 2)