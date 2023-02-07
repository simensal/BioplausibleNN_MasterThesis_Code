import numpy as np
import pandas as pd

def activation_function(x):
    """Calculates the activation of a node give an input

    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node
    """
    return (1 + np.e**-x)**-1

def tanh_function(x):
    """Calculates the activation of a node give an input

    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node
    """
    return (np.e**x - np.e**-x) / (np.e**x + np.e**-x)

def der_tanh_function(x):
    """Calculates the derivative of the activation function
    
    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node"""
    return 1 - tanh(x)**2

def normalize(dataset):
    """Normalizes the columns of the dataset

    Args:
        dataset (ndarray): Dataset to be normalized

    Returns:
        ndarray: Normalized dataset
    """
    return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))

def normalize_tanh(dataset):
    """Normalize dataset to be between -1 and 1
    
    Args:
        dataset (ndarray): Dataset to be normalized

    Returns:
        ndarray: Normalized dataset
    """
    return 2 * normalize(dataset) - 1

def MSE(y, y_pred):
    """Calculates the mean squared error between the predicted and the actual values
    """
    return np.mean((y - y_pred)**2)

def der_activation_function(x):
    """Calculates the derivative of the activation function
    
    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node"""
    return (np.e**-x)/((1+np.e**-x)**2)

def get_iris_data():
    """ Get the iris data and converts it to training and test data sets

    Returns:
        tuple: Training and test data sets
    """
    # Load data from iris and create training and test sets
    iris = pd.read_csv('./data/iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']).sample(frac=1)
    y = iris['label']
    X = iris.drop('label', axis=1)

    # Converting labels from string to int
    y = y.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # Splitting the data into training and test sets
    X_train = X.iloc[:110].to_numpy()
    X_test = X.iloc[110:].to_numpy()
    y_train = y.iloc[:110]
    y_test = y.iloc[110:]

    # Converting the labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train, X_test, y_train, y_test

tanh = np.vectorize(tanh_function)
der_tanh = np.vectorize(der_tanh_function)

activation = np.vectorize(activation_function)
der_activation = np.vectorize(der_activation_function)
