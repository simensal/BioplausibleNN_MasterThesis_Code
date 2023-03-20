import numpy as np
from pypapi import events, papi_high as high
from ANN import ANN as ANN
from PCN import PCN as PCN
from helpers import get_iris_data
import sys

def train_agent(agent, X_train, y_train, epochs, verbose, sample_interval):
    """
    Train the agent for a given number of epochs and return the agent and the
    number of flops performed during training.

    Parameters
    ----------
    agent : ANN or PCN
        The agent to be trained.
    X_train : np.ndarray
        The training set.
    y_train : np.ndarray
        The training labels.
    epochs : int
        The number of epochs to train the agent.
    verbose : bool
        Whether to print the progress of the training.
    sample_interval : int
        The interval at which to print the progress of the training.

    Returns
    -------
    agent : Learner
        The trained agent.
    flops : int
        The number of flops performed during training.
    """
    high.read_counters() # Resets counter for the agent
    for i in range(epochs): 
        agent.train(X_train, y_train, normalize_inputs=False)
        if verbose and (i+1) % sample_interval == 0:
            print(f'Finished epoch #{i+1}')

    return agent, high.read_counters()[0]

def main(argv):
    verbose = "-v" in argv or "--verbose" in argv

    # Loading data
    X_train, X_test, y_train, y_test = get_iris_data()

    # Setting up ANN and PCN agent
    ann = ANN(X_train.shape[1], [6], y_train.shape[1])
    pcn = PCN(X_train.shape[1], [6], y_train.shape[1])

    # Normalizing the dataset based on agent's scheme
    X_train_bp = ann.normalize(X_train)
    X_test_bp = ann.normalize(X_test)
    X_train_pcn = pcn.normalize(X_train)
    X_test_pcn = pcn.normalize(X_test)

    # Setting meta-params
    epochs = 30
    sample_interval = np.ceil(epochs/10).astype(int)

    # Starting flop counter
    high.start_counters([events.PAPI_DP_OPS])

    # Training ANN agent
    print("\n ------------------ ANN Training ------------------ \n")
    ann, ann_training_flops = train_agent(ann, X_train_bp, y_train, epochs, verbose, sample_interval)
    print(f'Flops performed over #{epochs} epochs: {ann_training_flops}')
    print(f'Average Flops per epoch: {ann_training_flops/epochs}')
    print("\n ---------------------------------------------------- \n")

    # Training PCN agent
    print("\n ------------------ PCN Training ------------------ \n")
    pcn, pcn_training_flops = train_agent(pcn, X_train_pcn, y_train, epochs, verbose, sample_interval)
    print(f'Flops performed over #{epochs} epochs: {pcn_training_flops}')
    print(f'Average Flops per epoch: {pcn_training_flops/epochs}')
    print("\n ---------------------------------------------------- \n")

    # Testing the ANN agent
    print('\n\n ------------------ ANN validation ------------------ \n')
    print('*** Training set ***')
    high.read_counters()
    _, ann_acc_train, _ = ann.test(X_train_bp, y_train, normalize_inputs=False)
    ann_vtrain_flops = high.read_counters()[0]
    print(f'Accuracy: {ann_acc_train}')
    print(f'Flops performed: {ann_vtrain_flops}')
    print('*******************')
    
    print('*** Test set ***')
    high.read_counters()
    _, ann_acc_test, _ = ann.test(X_test_bp, y_test, normalize_inputs=False)
    ann_vtest_flops = high.read_counters()[0]
    print(f'Accuracy: {ann_acc_test}')
    print(f'Flops performed: {ann_vtest_flops}')
    print('*******************')

    print('\n ---------------------------------------------------- \n')

    # Testing the PCN agent
    print('\n\n ------------------ PCN validation ------------------ \n')
    print('*** Training set ***')
    high.read_counters()
    _, pcn_acc_train, _ = pcn.test(X_train_pcn, y_train, normalize_inputs=False)
    pcn_vtrain_flops = high.read_counters()[0]
    print(f'Accuracy: {pcn_acc_train}')
    print(f'Flops performed: {pcn_vtrain_flops}')
    print('*******************')

    print('*** Test set ***')
    high.read_counters()
    _, pcn_acc_test, _ = pcn.test(X_test_pcn, y_test, normalize_inputs=False)
    pcn_vtest_flops = high.read_counters()[0]
    print(f'Accuracy: {pcn_acc_test}')
    print(f'Flops performed: {pcn_vtest_flops}') 
    print('*******************')

    print('\n ---------------------------------------------------- \n')

if __name__ == "__main__":
    main(sys.argv[1:])