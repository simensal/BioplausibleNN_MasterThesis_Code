import numpy as np
from learner import learner
from helpers import activation, der_activation, normalize


class ANN(learner):
    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01, activation=activation, der_activation=der_activation, normalize_function=normalize) -> None:
        """
        Takes as input
            hidden_layers : 1d array with number of neurons to include in each of the hidden layers
            features : int representing the number of features of the data set - corresponds to the first (input) layer
            outputs : int representing the number of possible outputs - and thereby number of nodes in the final (output) layer
        """

        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate,
                         activation=activation, der_activation=der_activation,
                         normalize_function=normalize_function
                         )

        combined_layers = np.array([features] + hidden_layers + [outputs])

        # Setting up the layers
        self.layers = []
        self.deltas = []
        for num in combined_layers:
            self.layers.append(np.zeros(num))
            self.deltas.append(np.zeros(num))

        # Setting up weights and layer inputs
        self.layer_inputs = []
        self.weights = []

        for i in range(len(combined_layers)-1):
            self.weights.append(np.random.rand(
                combined_layers[i], combined_layers[i+1]))
            self.layer_inputs.append(np.zeros(combined_layers[i]))

    def __predict__(self, sample) -> np.ndarray:
        """Predicts the label of a sample x

        Args:
            sample (ndarray): Array of floats which represents the first input layer of the network

        Returns:
            ndarray: Array of floats which represent the confidences of the classification for each category of the output
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

        return self.layers[-1]

    def backprops_error(self, solution) -> None:
        """Compares a prediction to a solution and calculates the error between the layer and the expected values
        (Propagates the deltas down the network from the output)

        Args:
            solution (ndarray): The solution to the sample fetched from the dataset
        """

        # Calculate error of output layer
        self.deltas[-1] = der_activation(self.layer_inputs[-1]) * \
            (self.layers[-1] - solution)

        # Iterate from outputlayer backwards and propagate error
        for i in range(len(self.weights) - 1, 0, -1):
            self.deltas[i] = der_activation(
                self.layer_inputs[i-1]) * np.dot(self.weights[i], self.deltas[i + 1])

    def update_network(self) -> None:
        """Calculates the margins of weights for the network given a traning example, and updates the weights accordingly
        """
        for i in range(len(self.weights)):
            margins = -self.learning_rate * \
                np.outer(self.layers[i], self.deltas[i + 1].T)
            self.weights[i] = self.weights[i] + margins

    def __validate__(self, samples, solutions) -> None:
        """Validates the input to the network

        Args:
            samples (ndarray(n,)): Set of samples for the network to make predictions on and train for
            solutions (ndarray(n,)): Set of respective solutions to the samples
        """
        assert len(samples) == len(solutions)
        assert len(samples[0]) == self.num_features
        assert len(solutions[0]) == self.num_outputs

    def train(self, samples, solutions) -> None:
        """Trains the network on a set of samples and respective solutions

        Args:
            samples (ndarray(n,)): Set of samples for the network to make predictions on and train for
            solutions (ndarray(n,)): Set of respective solutions to the samples
        """
        self.__validate__(samples, solutions)

        for sample, solution in zip(samples, solutions):
            self.__predict__(sample)
            self.backprops_error(solution)
            self.update_network()

    def test(self, samples, solutions, verbose=False, normalize=False) -> tuple:
        """Test the perfomance of the network on a dataset

        Args:
            samples ((ndarray(n,))): Set of samples
            solutions (ndarray(n,)): Set of solutions to respective examples

        Returns:
            error (ndarray(n,)) : Set of squared errors for each sample
            predictions (ndarray(n,)) : Predictions for each sample (calculated by argmax)
        """
        self.__validate__(samples, solutions)
        return super().test(samples, solutions, normalize_inputs=normalize, verbose=verbose)
