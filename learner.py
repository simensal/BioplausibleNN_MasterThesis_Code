import numpy as np

from helpers import activation, der_activation, normalize, MSE


class learner:

    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01, activation=activation, der_activation=der_activation, normalize_function=normalize) -> None:
        self.num_features = features
        self.num_outputs = outputs
        self.num_layers = len(hidden_layers) + 2
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        self.activation = activation
        self.der_activation = der_activation
        self.normalize = normalize_function

    def __predict__(self, sample) -> np.ndarray:
        """
        Predicts the output of a sample
        Takes as input a sample and returns the output of the network as ndarray
        """
        pass

    def reset(self) -> None:
        """
        Resets the learner to the state at initialization
        """
        self.__init__(self.num_features, self.hidden_layers,
                      self.num_outputs, self.learning_rate)

    def train(self, samples, solutions, normalize_inputs=False) -> None:
        pass

    def test(self, samples, solutions, verbose=False, normalize_inputs=False) -> tuple:
        """
        Test the learner on a set of samples and solutions
        """
        # Normalize dataset
        if normalize_inputs:
            samples = normalize(samples)

        # Setting up lists for predictions and errors
        predictions = []
        error = []

        # Predicting and calculating error
        for sample, solution in zip(samples, solutions):
            prediction = self.__predict__(sample)
            if verbose:
                print("Prediction: ", prediction, end="  ")
                print("Solution: ", solution)
            predictions.append(prediction)
            # Change this to use MSE
            error.append(np.sum(np.square(solution - prediction)))

        # Calculate accuracy
        accuracy = np.divide(np.equal(np.argmax(predictions, axis=1), np.argmax(
            solutions, axis=1)).sum(), len(solutions))
        return predictions, accuracy, error

    # def normalize(self, samples) -> np.ndarray:
    #     """
    #     Normalizes the dataset
    #     """
    #     return self.normalize(samples)

    def normalize(self, X_train, X_test) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalizes the dataset
        """
        return self.normalize(X_train, X_test)