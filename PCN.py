import numpy as np

from helpers import tanh, der_tanh, normalize_tanh, soft_max, der_soft_max
from learner import learner


class PCN(learner):
    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01, max_iter=100, activation=tanh, der_activation=der_tanh, normalize_function=normalize_tanh, convergence_tolerance=1e-2, variance=4):
        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate,
                         activation=activation, der_activation=der_activation,
                         normalize_function=normalize_function)

        self.variances = 1
        self.max_iter = max_iter
        self.convergence_tolerance = convergence_tolerance

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
                self.weights.append(np.random.uniform(-0.1, 0.1, (
                    self.num_features, self.hidden_layers[i])))
            elif i == self.num_layers - 2:
                self.weights.append(np.random.uniform(-0.1, 0.1, (
                    self.hidden_layers[i-1], self.num_outputs)))
            else:
                self.weights.append(np.random.uniform(-0.1, 0.1, (
                    self.hidden_layers[i-1], self.hidden_layers[i])))

        # Creating variances for error layers
        self.variance_matrix = []
        for i in range(self.num_layers - 1):
            if i == self.num_layers - 2:
                self.variance_matrix.append(
                    np.ones((self.num_outputs))*variance)
            else:
                self.variance_matrix.append(
                    np.ones((self.hidden_layers[i]))*variance)

    def __check_convergence__(self, prev_state, curr_state):
        """
        Checks if the state of the network has converged within given tolerance

        returns: (bool) True if converged, False otherwise
        """
        for i in range(self.num_layers):
            if not np.allclose(prev_state[i], curr_state[i], atol=self.convergence_tolerance):
                return False
        return True

    def __predict__(self, sample, max_iter=None, output_clamped=False):
        """
        Take a sample as input and converges to fixed point equilibrium

        returns: (ndarray) predicted output to the given input sample
        """

        if max_iter is None:
            max_iter = self.max_iter    

        output_update_mask = [not output_clamped for _ in range(self.num_outputs)]
        input_clamp = np.isnan(sample)

        # Set input layer equal to the normalized input sample
        self.layers[0] = np.nan_to_num(sample)

        # Converge towards equilibrium state
        t = 0
        curr_state = self.layers.copy()
        while True:
            t += 1

            # Updating the input layer based on feedback errors
            self.layers[0][input_clamp] = (np.dot(self.error_layers[0], self.weights[0].T) * self.der_activation(self.layers[0]))[input_clamp]

            # Perform iteration
            for i in range(self.num_layers - 1):

                # Update error layer
                self.error_layers[i] = np.divide(
                    (self.layers[i+1] - np.dot(self.activation(self.layers[i]), self.weights[i])), self.variance_matrix[i])
                # self.error_layers[i] = np.divide((self.layers[i+1] - self.activation(np.dot(self.layers[i], self.weights[i]))), self.variance_matrix[i])

                # Update activation layer based on autoerror and upstream error
                if i == len(self.layers) - 2:  # Output layer
                    updated_values = (self.layers[i+1] - self.error_layers[i])
                    self.layers[i+1][output_update_mask] = updated_values[output_update_mask]
                else:  # Intermediary layers
                    self.layers[i+1] = self.layers[i+1] - self.error_layers[i] + np.dot(
                        self.error_layers[i+1], self.weights[i+1].T) * self.der_activation(self.layers[i+1])

            # Convergence condition
            if self.__check_convergence__(curr_state, self.layers) or t >= max_iter:
                break

            # Update current prediction
            curr_state = self.layers.copy()

        return self.layers[-1].copy(), self.layers[0].copy(), t >= max_iter

    def train(self, samples, solutions, normalize_inputs=True) -> None:
        """
        Train the network based on the provided samples and solutions
        """
        # Normalize samples
        if normalize_inputs:
            samples = self.normalize(samples)

        exceeded_timelimit = []

        # Loop over all samples and learn from sample
        for sample, solution in zip(samples, solutions):
            # Clamp output and input
            self.layers[-1] = solution.astype(np.float64)
            _, _, exceeded = self.__predict__(sample, output_clamped=True)
            exceeded_timelimit.append(exceeded)

            # Update weights based on residual error in network after convergence (eq. 2.19 in (Whittington, Bogacz - 2017)))
            for i in range(self.num_layers - 1):
                self.weights[i] = self.weights[i] + \
                    self.learning_rate * \
                    np.outer(self.activation(
                        self.layers[i]), self.error_layers[i])

        return exceeded_timelimit

    def test(self, samples, solutions, verbose=False, normalize_inputs=True) -> tuple:
        """
        Test the learner on a set of samples and solutions

        Args:
            samples: (ndarray) samples to test on
            solutions: (ndarray) solutions to test on
            verbose: (bool = False) print predictions and solutions
            normalize_inputs: (bool = True) normalize inputs before testing

        returns: (tuple) (predictions, accuracy, error)
        """
        # Normalize dataset
        if normalize_inputs:
            samples = self.normalize(samples)

        # Setting up lists for predictions and errors
        predictions = []
        error = []
        exceeded_timelimit = []

        # Predicting and calculating error
        for sample, solution in zip(samples, solutions):
            prediction, _, exceeded = self.__predict__(sample)
            exceeded_timelimit.append(exceeded)
            if verbose:
                print("Prediction: ", prediction, end="  ")
                print("Solution: ", solution)
            predictions.append(prediction)
            # Change this to use MSE
            error.append(np.sum(np.square(solution - prediction)))

        # Calculate accuracy
        accuracy = np.divide(np.equal(np.argmax(predictions, axis=1), np.argmax(
            solutions, axis=1)).sum(), len(solutions))
        return predictions, accuracy, error  # , exceeded_timelimit


class PCN_soft(PCN):
    """PCN class with softmax layer at the end"""
    def __init__(self, 
        features:int, 
        hidden_layers:list, 
        outputs:int, 
        learning_rate:float=0.01, 
        max_iter:int=100, 
        activation=tanh, 
        der_activation=der_tanh, 
        normalize_function=normalize_tanh, 
        convergence_tolerance:float=5e-3, 
        variance:int=4):
        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate,
                            max_iter=max_iter, activation=activation,
                            der_activation=der_activation,
                            normalize_function=normalize_function,
                            variance=variance,
                            convergence_tolerance=convergence_tolerance,
                         )

        # TODO: Fix the new layer here for error and activation
        self.error_layers.append(np.zeros(outputs))
        self.layers.append(np.zeros(outputs))
        self.weights.append(np.ones((self.num_outputs, self.num_outputs)))
        # self.num_layers += 1

    def __predict__(self, sample, max_iter=None, output_clamped=False, output_clamp = None):
        """
        Take a sample as input and converges to fixed point equilibrium

        returns: (ndarray) predicted output to the given input sample
        """

        if max_iter is None:
            max_iter = self.max_iter
        if output_clamp is None:
            output_clamp = [not output_clamped for _ in range(self.num_outputs)]

        # Set input layer equal to the normalized input sample
        self.layers[0] = sample

        # Converge towards equilibrium state
        t = 0
        curr_state = self.layers.copy()
        while True:
            t += 1

            # Perform iteration
            for i in range(self.num_layers - 1):

                self.error_layers[i] = np.divide(
                        (self.layers[i+1] - np.dot(self.activation(self.layers[i]), self.weights[i])), self.variance_matrix[i])

                self.layers[i+1] = self.layers[i+1] - self.error_layers[i] + np.dot(
                    self.error_layers[i+1], self.weights[i+1].T) * self.der_activation(self.layers[i+1])
            
            # Softmax layer
            self.error_layers[-1] = np.divide(self.layers[-1] - soft_max(self.layers[-2]), 4)
            self.layers[-1][output_clamp] = (self.layers[-1] - self.error_layers[-1])[output_clamp]

            # self.layers[-1][output_clamp] = (self.layers[-1] - self.error_layers[-1])[output_clamp]

            # layers[-1][[True, False, True]] = np.random.rand(layers[-1].shape[0])[[True, False, True]]

            # Convergence condition
            if self.__check_convergence__(curr_state, self.layers) or t >= max_iter:
                break

            # Update current prediction
            curr_state = self.layers.copy()

        return self.layers[-1].copy(), t >= max_iter

class PCN_tolerance(PCN):

    def __init__(self, 
        features:int, 
        hidden_layers:list, 
        outputs:int, 
        learning_rate:float=0.01, 
        max_iter:int=100, 
        activation=tanh, 
        der_activation=der_tanh, 
        normalize_function=normalize_tanh, 
        convergence_tolerance:float=5e-3, 
        variance:int=4):
        super().__init__(features=features, hidden_layers=hidden_layers,
                         outputs=outputs, learning_rate=learning_rate,
                            max_iter=max_iter, activation=activation,
                            der_activation=der_activation,
                            normalize_function=normalize_function,
                            variance=variance,
                            convergence_tolerance=convergence_tolerance,
                         )

    def __check_convergence__(self, prev_state, curr_state):
        """Checks if the change in the network is smaller than the given tolerance in total percentage change of network energy
        
        Args:
            prev_state: (ndarray) previous state of the network
            curr_state: (ndarray) current state of the network
        
        returns: (bool) True if converged, False otherwise
        """
        def __get_energy__(state):
            energy = 0
            for i in range(self.num_layers - 1):
                # energy += np.sum(np.square(state[i+1] - np.dot(self.activation(state[i]), self.weights[i])))
                energy += np.sum(np.abs(state[i]))
            return energy
        
        prev_energy = __get_energy__(prev_state)
        curr_energy = __get_energy__(curr_state)
        return np.divide(np.abs(prev_energy - curr_energy), prev_energy) < self.convergence_tolerance

