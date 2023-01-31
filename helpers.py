import numpy as np

def activation_function(x):
    """Calculates the activation of a node give an input

    Args:
        x (integer: Scalar) :  Number representing the sum of the input to the give node
    """
    return (1 + np.e**-x)**-1


def normalize(dataset):
    """Normalizes the dataset

    Args:
        dataset (ndarray): Dataset to be normalized

    Returns:
        ndarray: Normalized dataset
    """
    return (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

def der_activation_function(x):
    """Calculates the derivative of the activation function
    
    Args:
        x (integer: Scalar) :  Number representing the sum of the input to the give node"""
    return (np.e**-x)/((1+np.e**-x)**2)


activation = np.vectorize(activation_function)
der_activation = np.vectorize(der_activation_function)
