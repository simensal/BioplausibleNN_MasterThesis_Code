import numpy as np

from helpers import activation, der_activation

class ANN:
    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01):
        """
        Takes as input
            hidden_layers : 1d array with number of neurons to include in each of the hidden layers
            features : int representing the number of features of the data set - corresponds to the first (input) layer
            outputs : int representing the number of possible outputs - and thereby number of nodes in the final (output) layer
        """

        # Meta data
        self.num_layers = len(hidden_layers) + 2
        self.num_outputs = outputs
        self.num_features = features
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        combined_layers = np.array([features] + hidden_layers + [outputs])

        # Setting up the layers
        self.layers = []
        self.deltas = []
        for num in combined_layers:
            self.layers.append([0 for _ in range(num)])
            self.deltas.append([0 for _ in range(num)])

        # Setting up weights and layer inputs
        self.layer_inputs = []
        self.weights = []
        #self.margins = []
        for i in range(len(combined_layers)-1):
            self.weights.append(np.random.rand(
                combined_layers[i], combined_layers[i+1]))
            #self.margins.append(np.zeros(np.shape((combined_layers[i], combined_layers[i+1]))))
            self.layer_inputs.append([0 for _ in range(combined_layers[i])])

    def reset(self):
        self.__init__(self.num_features, self.hidden_layers, self.num_outputs)

    def forward_pass(self, sample):
        """Fixes the neural network to a sample and generates the prediction at the output layer

        Args:
            sample (ndarray): Array of floats which represents the value of the input sample
        """
        self.layers[0] = sample

        # Iterate through each layer and calculate activation based on
        for i in range(1, self.num_layers):
            # Fetch the activation and weight of the previous layers
            prev_activation = self.layers[i-1]
            weights = self.weights[i-1]

            # Calculate inputs of the current nodes based on activation of previous and weights between them
            self.layer_inputs[i-1] = np.dot(weights.T, prev_activation)

            # Update current layer with activation of inputs
            self.layers[i] = activation(self.layer_inputs[i-1])

    def predict(self, sample):
        """Predicts the label of a sample x

        Args:
            sample (ndarray): Array of integers which represents the first input layer of the network

        Returns:
            ndarray: Array of floats which represent the confidences of the classification for each category of the output
        """
        self.forward_pass(sample)
        return self.layers[-1]

    def backprops_error(self, solution):
        """Compares a prediction to a solution and calculates the error between the layer and the expected values
        (Propagates the deltas down the network from the output)

        Args:
            prediction (ndarray): Prediction, output of the network for some sample x
            solution (ndarray): The solution to the sample fetched from the dataset
        """
        
        # Calculate error of output layer
        self.deltas[-1] = der_activation(self.layer_inputs[-1]) * (self.layers[-1] - solution)

        # Iterate from outputlayer backwards and propagate error
        for i in range(len(self.weights) - 1, 0, -1):
            self.deltas[i] = der_activation(self.layer_inputs[i-1]) * np.dot(self.weights[i], self.deltas[i + 1])

    def update_network(self, solution):
        """Calculates the margins of weights for the network given a traning example, and updates the weights accordingly
        """
        for i in range(len(self.weights)):
            margins = -self.learning_rate * np.outer(self.layers[i], self.deltas[i + 1].T)
            self.weights[i] = self.weights[i] + margins

    def train_sample(self, sample, solution): 
        """Trains the network on a single sample

        Args:
            sample (ndarray(4,)): Sample to make prediction on
            solution (ndarray(3,)): Solution to the respective sample passed
        """
        assert len(sample) == self.num_features
        assert len(solution) == self.num_outputs
        
        self.predict(sample)
        self.backprops_error(solution)
        self.update_network(solution)

    def train(self, samples, solutions):
        """Trains the network on a set of samples and respective solutions

        Args:
            samples (ndarray(n,)): Set of samples for the network to make predictions on and train for
            solutions (ndarray(n,)): Set of respective solutions to the samples
        """
        assert len(samples) == len(solutions)

        for sample, solution in zip(samples, solutions):
            self.train_sample(sample, solution)

    def test_sample(self, sample, solution):
        """Tests a single sample and returns the squared error

        Args:
            sample (ndarray(num_features,)): Sample to predict and find error for
            solution (ndarray(num_outputs)): Solutions to respective example

        Returns:
            (float): Squared error over solution
            pred (ndarray(#classes,)) : Prediction over the classes
        """
        assert len(sample) == self.num_features
        assert len(solution) == self.num_outputs

        pred = self.predict(sample)

        return np.sum((pred - solution)**2), pred

    def test(self, samples, solutions):
        """Test the perfomance of the network on a dataset

        Args:
            samples ((ndarray(n,))): Set of samples
            solutions (ndarray(n,)): Set of solutions to respective examples

        Returns:
            error (ndarray(n,)) : Set of squared errors for each sample
            predictions (ndarray(n,)) : Predictions for each sample (calculated by argmax)
        """
        assert len(samples) == len(solutions)

        error = np.array([])
        predictions = np.array([])

        for sample, solution in zip(samples, solutions):
            sample_error, sample_prediction = self.test_sample(sample, solution)
            error = np.append(error, sample_error)
            predictions = np.append(predictions, np.argmax(sample_prediction))

        return error, predictions
