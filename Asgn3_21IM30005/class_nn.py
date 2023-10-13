import numpy as np

class NN():
    def __init__(self, input_layer_size = 2, hidden_layer_size = 2, output_layer_size = 2, hidden_layers = 1, lr = 0.1, epoches = 1000):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.number_of_epoches = epoches

        # initialize the weights by random numbers
        self.W = [np.random.randn(self.input_layer_size, self.hidden_layer_size)]
        for layer in range(hidden_layers-1):
            self.W.append(np.random.randn(self.hidden_layer_size, self.hidden_layer_size))
        self.W.append(np.random.randn(self.hidden_layer_size, self.output_layer_size))

    def activation(self, x): # sigmoid function
        return (1/(1+np.exp(-x)))
    
    def forward(self, X, less_steps = False): # forward prapogation
        # X- input
        input_list = X

        if less_steps == False: # for full forwward propagation to get final output
            less_steps = self.hidden_layers + 1 
        # if a value is given to "steps" parameter, then we perform forward propagation to that step 

        for step in range(less_steps):
            output_list = np.dot(input_list, self.W[step])
            output_list = [self.activation(neuron) for neuron in output_list]
            input_list = output_list
        return input_list # since input_list = output_list at the end of the iterations and also we can use it for our loop in backward()
    
    def backward(self, X, y): # backward propagation
        # X- input, y- output
        # output_list = self.forward(X)
        # total error = np.sum((y - output_list)**2/2)
        hidden_layers = self.hidden_layers
        # first back propagation for last layer
        # error = (output)(1-output)(actual-output)
        layer_error = np.array(self.forward(X)) * (1 - np.array(self.forward(X))) * (np.array(y) - np.array(self.forward(X)))
        # Weight updation- W = W + alpha * (H * ((y-OP) * (OP * (1 - OP))))
        self.W[-1] += self.lr * (np.dot(np.matrix(self.forward(X,hidden_layers)[0]).transpose(), layer_error))

        # there are hidden_layers + 1 number of weight matrices, hidden_layers - 1 if we calculate first and last one out of loop
        for step in range(2,hidden_layers+2):
            layer_error = np.array(self.forward(X, hidden_layers+2-step)) * (1 - np.array(self.forward(X, hidden_layers+2-step))) * np.dot(self.W[-step+1],np.matrix(layer_error).transpose())
            self.W[-step] += self.lr * (np.dot(np.matrix(self.forward(X,hidden_layers+1-step)[0]).transpose(), layer_error))


    # training: update the weights for the given number of epoches
    def train(self, X, y):
        for _ in range(self.number_of_epoches):
            self.backward(X, y)
    
    def total_error(self,X,y):
        return np.sum((y - self.forward(X))**2/2)  