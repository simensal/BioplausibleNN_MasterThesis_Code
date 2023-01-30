import numpy as np

from helpers import activation, der_activation

class learner:

    def __init__(self, features, hidden_layers, outputs, learning_rate=0.01) -> None:
        self.num_features = features
        self.num_outputs = outputs
        self.num_layers = len(hidden_layers) + 2 
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

    def reset(self):
        """Resets the learner to the state at initialization"""
        self.__init__(self.num_features, self.hidden_layers, self.num_outputs, self.learning_rate)

    def train():
        pass

    def test(self, samples, solutions):
        """
        Test the learner on a set of samples and solutions
        """
        predictions = []
        for sample, solution in zip(samples, solutions):
            prediction = self.predict(sample)
            predictions.append(prediction)
        
        # Calculate accuracy
        accuracy = np.mean(np.equal(predictions, solutions))
        return predictions, accuracy

    