import numpy as np

from helpers import activation, der_activation, normalize
from learner import learner


class PCN(learner):
    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01, max_iter=1000, activation=activation, der_activation=der_activation, normalize_function=normalize):
        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate,
                         activation=activation, der_activation=der_activation,
                         normalize_function=normalize_function)

        self.variances = 1
        self.max_iter = max_iter

        # Create layers with activation and error nodes
        self.layers = []
        self.error_layers = []
        for i in range(self.num_layers):
            # Input layer
            if i == 0:
                self.layers.append(np.zeros(self.num_features))
                # self.error_layers.append(np.zeros(self.num_features))

            # Output layer
            elif i == self.num_layers - 1:
                self.layers.append(np.zeros(self.num_outputs))
                self.error_layers.append(np.zeros(self.num_outputs))

            # Hidden layers
            else:
                self.layers.append(np.zeros(self.hidden_layers[i-1]))
                self.error_layers.append(np.zeros(self.hidden_layers[i-1]))

        # Create weights
        self.weights = []
        for i in range(self.num_layers - 1):
            if i == 0:
                self.weights.append(np.random.rand(
                    self.num_features, self.hidden_layers[i]))
            elif i == self.num_layers - 2:
                self.weights.append(np.random.rand(
                    self.hidden_layers[i-1], self.num_outputs))
            else:
                self.weights.append(np.random.rand(
                    self.hidden_layers[i-1], self.hidden_layers[i]))

        # Creating variances for error layers
        self.variance_matrix = []
        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                self.variance_matrix.append(np.ones((self.num_outputs))*8)
            else:
                self.variance_matrix.append(np.ones((self.hidden_layers[i]))*8)

    def __check_convergence__(self, prev_state, curr_state):
        """
        Checks if the state of the network has converged within given tolerance

        returns: (bool) True if converged, False otherwise
        """
        for i in range(self.num_layers):
            if not np.allclose(prev_state[i], curr_state[i], atol=1e-3):
                return False
        return True

    def __predict__(self, sample, max_iter=None, output_clamped=False):
        """
        Take a sample as input and converges to fixed point equilibrium

        returns: (ndarray) predicted output to the given input sample
        """

        if max_iter is None:
            max_iter = self.max_iter

        # Set input layer equal to the normalized input sample
        self.layers[0] = sample

        # Converge towards equilibrium state
        t = 0
        curr_state = self.layers.copy()
        while True:
            t += 1

            # Perform iteration
            for i in range(self.num_layers - 1):

                # Update error layer
                self.error_layers[i] = np.divide(
                    (self.layers[i+1] - np.dot(activation(self.layers[i]), self.weights[i])), self.variance_matrix[i])
                # self.error_layers[i] = np.divide((self.layers[i+1] - activation(np.dot(self.layers[i], self.weights[i]))), self.variance_matrix[i])

                # Update activation layer based on autoerror and upstream error
                if i == len(self.layers) - 2:  # Output layer
                    if not output_clamped:
                        self.layers[i+1] = self.layers[i+1] - \
                            self.error_layers[i]
                else:  # Intermediary layers
                    self.layers[i+1] = self.layers[i+1] - self.error_layers[i] + np.dot(
                        self.error_layers[i+1], self.weights[i+1].T) * der_activation(self.layers[i+1])

            # Convergence condition
            if self.__check_convergence__(curr_state, self.layers) or t >= max_iter:
                break

            # Update current prediction
            curr_state = self.layers.copy()

        return self.layers[-1]

    def train(self, samples, solutions) -> None:
        """
        Train the network based on the provided samples and solutions
        """
        # Normalize samples
        samples = normalize(samples)

        # Loop over all samples and learn from sample
        for sample, solution in zip(samples, solutions):
            # Clamp output and input
            self.layers[-1] = solution
            self.__predict__(sample, output_clamped=True)

            # Update weights based on residual error in network after convergence (eq. 2.19 in (Whittington, Bogacz - 2017)))
            for i in range(self.num_layers - 1):
                # activations = self.layers[i]
                activations = activation(self.layers[i])
                errors = self.error_layers[i]
                self.weights[i] = self.weights[i] + \
                    self.learning_rate * np.outer(activations, errors)

# WIP


class PCN_batcher(PCN):

    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01, max_iter=1000, batch_size=10):
        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate)
        self.batch_size = batch_size

    def train(self, samples, solutions, batch_size=None) -> None:
        """
        Train the network based on the provided samples and solutions
        """
        # Normalize samples
        samples = normalize(samples)

        # Set batch size
        if batch_size is None:
            batch_size = self.batch_size

        # Setting up margins for batch
        margins = []

        # Loop over all samples and learn from sample
        for i, tup in enumerate(zip(samples, solutions)):
            sample, solution = tup

            # Clamp output and input
            self.layers[-1] = solution
            self.__predict__(sample, output_clamped=True)

            # Record residual for batching (eq. 2.19 in (Whittington, Bogacz - 2017)))
            sample_margin = []
            for j in range(self.num_layers - 1):
                activations = activation(self.layers[j])
                errors = self.error_layers[j]
                sample_margin.append(np.outer(activations, errors))
            margins.append(sample_margin)

            # Update weights if on batch size
            if i % batch_size == 0:
                self.weights = self.weights + self.learning_rate * \
                    np.mean(margins, axis=0, dtype=object)
                margins = []
