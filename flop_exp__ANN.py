import numpy as np
from pypapi import events, papi_high as high
from ANN import ANN as ANN
from helpers import get_iris_data
import sys

def main(argv):

    verbose = "-v" in argv or "--verbose" in argv

    # Get the data
    X_train, X_test, y_train, y_test = get_iris_data()

    # Initialize the ANN
    ann = ANN(X_train.shape[1], [6], y_train.shape[1])

    # Normalizing the dataset based on agent's scheme
    X_train = ann.normalize(X_train)
    X_test = ann.normalize(X_test)

    # Setting meta-params
    epochs = 30
    sample_interval = np.ceil(epochs/10).astype(int)
    
    # Starting flop counter
    high.start_counters([events.PAPI_DP_OPS])

    # Training the agent
    print("Training the agent...")
    for i in range(epochs): 
        ann.train(X_train, y_train, normalize_inputs=False)
        if verbose and (i+1) % sample_interval == 0:
            print(f'Finished epoch #{i+1}')

    # Reading and resetting flop counter
    pcn_training_flops = high.read_counters()[0]
    print(f'Flops performed over #{epochs} epochs: {pcn_training_flops}')
    print(f'Average Flops per epoch: {pcn_training_flops/epochs}')

    # Testing the agent
    print('\n\n ------------------ Validation set ------------------ \n')
    _, acc_test, _ = ann.test(X_test, y_test, normalize_inputs=False)
    print(f'Accuracy: {acc_test}')
    print(f'Flops performed during test validation: {high.read_counters()[0]}')
    print("\n---------------------------------------------------\n")

    # Printing the flops
    print('\n ------------------ Training set ------------------ \n')
    _, acc_train, _ = ann.test(X_train, y_train, normalize_inputs=False)
    print(f'Accuracy: {acc_train}')
    print(f'Flops performed during training validation: {high.read_counters()[0]}')
    print("\n---------------------------------------------------\n")
    high.stop_counters()


if __name__ == "__main__":
    main(sys.argv[1:])