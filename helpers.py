import numpy as np
import pandas as pd
import cProfile, pstats

def activation_function(x):
    """Calculates the activation of a node give an input

    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node
    """
    return (1 + np.e**-x)**-1


def der_activation_function(x):
    """Calculates the derivative of the activation function

    Args:
        x (float: Scalar) :  Number representing the sum of the input to the give node"""
    return (np.e**-x)/((1+np.e**-x)**2)

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
    return 1 - tanh_function(x)**2


# def normalize(dataset):
#     """Normalizes the columns of the dataset

#     Args:
#         dataset (ndarray): Dataset to be normalized

#     Returns:
#         ndarray: Normalized dataset
#     """
#     return (dataset - np.min(dataset, axis=0)) / (np.max(dataset, axis=0) - np.min(dataset, axis=0))


# def normalize_tanh(dataset):
#     """Normalize dataset to be between -1 and 1

#     Args:
#         dataset (ndarray): Dataset to be normalized

#     Returns:
#         ndarray: Normalized dataset
#     """
#     return 2 * normalize(dataset) - 1

def normalize(X_train, X_test): 
    """Normalizes the columns of the dataset

    Args:
        dataset (ndarray): Dataset to be normalized

    Returns:
        ndarray: Normalized dataset
    """

    X_train_min = np.min(X_train, axis=0)
    X_train_max = np.max(X_train, axis=0)

    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

    return X_train, X_test

def normalize_tanh(X_train, X_test):
    """Normalize dataset to be between -1 and 1

    Args:
        dataset (ndarray): Dataset to be normalized

    Returns:
        ndarray: Normalized dataset
    """

    X_train_min = np.min(X_train, axis=0)
    X_train_max = np.max(X_train, axis=0)

    X_train = 2 * ((X_train - X_train_min) / (X_train_max - X_train_min)) - 1
    X_test = 2 * ((X_test - X_train_min) / (X_train_max - X_train_min)) - 1

    return X_train, X_test

tanh = np.vectorize(tanh_function)
der_tanh = np.vectorize(der_tanh_function)

activation = np.vectorize(activation_function)
der_activation = np.vectorize(der_activation_function)


def soft_max(layer):
    """Calculates the activation of a node give an input

    Args:
        x (ndarray: scalars) :  array representing the final layer of the network
    """
    return np.exp(layer)/np.sum(np.exp(layer), axis=0)


def der_soft_max(layer):
    """Calculates the derivative of the activation function

    Args:
        layers (ndarray: Scalars) :  array representing the soft max of the final layer of the network

    Returns:
        ndarray: Derivative of the soft max for for the final layer of the network"""
    return np.diagflat(layer) - np.outer(layer, layer)


def MSE(y, y_pred):
    """Calculates the mean squared error between the predicted and the actual values
    """
    return np.mean((y - y_pred)**2)

### Data helpers ###


def get_data(path, split=0.8, sep=',') -> tuple:
    """Gets the data from the given path

    Args:
        path (str): Path to the data file

    Returns:
        tuple: Training and test data sets
    """

    df = pd.read_csv(path, header=None, sep=sep).sample(frac=1)

    # Splitting the data into training and test sets
    X_train = df.iloc[:int(len(df)*split)].drop(0, axis=1).to_numpy()
    X_test = df.iloc[int(len(df)*split):].drop(0, axis=1).to_numpy()
    y_train = df.iloc[:int(len(df)*split)][0]
    y_test = df.iloc[int(len(df)*split):][0]

    # Converting the labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train, X_test, y_train, y_test


def get_yeast_data():
    """ Get the yeast data and converts it to training and test data sets

    Returns:
        tuple: Training and test data sets (X_train, X_test, y_train, y_test)
    """

    df = pd.read_csv('./data/yeast/yeast.data', header=None,
                     sep='\s+').sample(frac=1).drop(0, axis=1)

    # Splitting the data into training and test sets
    X_train = df.iloc[:int(len(df)*0.8)].drop(9, axis=1).to_numpy()
    X_test = df.iloc[int(len(df)*0.8):].drop(9, axis=1).to_numpy()
    y_train = df.iloc[:int(len(df)*0.8)][9]
    y_test = df.iloc[int(len(df)*0.8):][9]

    # Converting the labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train, X_test, y_train, y_test


def get_iris_data():
    """ Get the iris data and converts it to training and test data sets

    Returns:
        tuple: Training and test data sets
    """
    # Load data from iris and create training and test sets
    iris = pd.read_csv('./data/iris.data', names=[
                       'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']).sample(frac=1)
    y = iris['label']
    X = iris.drop('label', axis=1)

    # Converting labels from string to int
    y = y.replace(
        {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # Splitting the data into training and test sets
    X_train = X.iloc[:110].to_numpy()
    X_test = X.iloc[110:].to_numpy()
    y_train = y.iloc[:110]
    y_test = y.iloc[110:]

    # Converting the labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train, X_test, y_train, y_test


def get_mnist_data(num_samples=1000, test_split=0.2):
    """ Get the mnist data and converts it to training and test data sets

    Returns:
        tuple: Training and test data sets (X_train, X_test, y_train, y_test)
    """

    # Load data from mnist and create training and test sets
    mnist = pd.read_csv('./data/mnist/mnist_train.csv')/256
    y_train = mnist['label'][:int(num_samples*(1-test_split))]
    X_train = mnist.drop('label', axis=1)[:int(num_samples*(1-test_split))]

    # mnist = pd.read_csv('./data/mnist/mnist_test.csv')/256
    y_test = mnist['label'][:int(num_samples*test_split)]
    X_test = mnist.drop('label', axis=1)[:int(num_samples*test_split)]

    # Converting the labels to one-hot encoding
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values

    return X_train.to_numpy(), X_test.to_numpy(), y_train, y_test

def mask_data(dataset, fraction):
    """Masks the data with the given fraction

    Args:
        dataset (ndarray): Dataset to be masked
        fraction (float): Fraction of the data to be masked

    Returns:
        ndarray: Masked dataset
    """
    mask = np.random.choice([True, False], size=dataset.shape, p=[
                            fraction, 1-fraction])
    sparse_dataset = dataset.copy()
    sparse_dataset[mask] = np.nan
    return sparse_dataset

def profile_code(code_string,sortby='calls',frac=0.1):
    pr = cProfile.Profile()
    pr.enable()
    pr.run(code_string)
    pr.disable()
    ps = pstats.Stats(pr)
    ps.strip_dirs()  # Removes all path info....else UGLY printouts.  This MUST precede sort_stats.
    ps.sort_stats(sortby)
    ps.print_stats(frac)  # print first frac(tion) of ALL the results.

class settings:

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        match dataset:
            case 'iris':
                self.epochs = 30
                self.hidden_layers = [6]
                self.lr = 0.01
                self.conv_tol = 0.01
                self.stopping_acc = 0.80
            case 'wine':
                self.epochs = 30
                self.hidden_layers = [15]
                self.lr = 0.01
                self.conv_tol = 0.01
                self.stopping_acc = 0.85
            case 'mnist':
                self.epochs = 30
                self.hidden_layers = [500, 500]
                self.lr = 0.01
                self.conv_tol = 0.01
                self.stopping_acc = 0.75 # try with 0.75 if timeouts again
            case 'yeast':
                self.epochs = 30
                self.hidden_layers = [15]
                self.lr = 0.01
                self.conv_tol = 0.01
                self.stopping_acc = 0.80
            case _:
                raise ValueError(f'Unknown dataset: {dataset}')

    def get_data(self):
        match self.dataset:
            case 'iris':
                return get_iris_data()
            case 'wine':
                return get_data('./data/wine/wine.data')
            case 'mnist':
                return get_mnist_data()
            case 'yeast':
                return get_yeast_data()
            case _:
                raise ValueError(f'Unknown dataset: {self.dataset}')
