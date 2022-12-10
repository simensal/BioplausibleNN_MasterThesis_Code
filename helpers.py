import numpy as np

def activation_function(x):
    """Calculates the activation of a node give an input

    Args:
        x (integer: Scalar) :  Number representing the sum of the input to the give node
    """
    return (1 + np.e**-x)**-1


def der_activation_function(x):

    return (np.e**-x)/((1+np.e**-x)**2)


activation = np.vectorize(activation_function)
der_activation = np.vectorize(der_activation_function)
