import numpy as np
import pandas as pd
from pypapi import events, papi_high as high
from ANN import ANN as ANN
from PCN import PCN as PCN
from helpers import settings as Settings
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
    save = "-s" in argv or "--save" in argv
    normalize = "-n" in argv or "--normalize" in argv

    n_agents = 10
    if '-n_agents' in argv:
        n_agents = int(argv[argv.index('-n_agents') + 1])

    # Setting up settings for the experiment
    dataset = argv[argv.index('-d') + 1]
    settings = Settings(dataset)

    hidden_layers = settings.hidden_layers

    # Loading data
    X_train, X_test, y_train, y_test = settings.get_data()

    # Setting up ANN and PCN agent
    anns = [ANN(X_train.shape[1], hidden_layers, y_train.shape[1]) for _ in range(n_agents)]
    pcns = [PCN(X_train.shape[1], hidden_layers, y_train.shape[1]) for _ in range(n_agents)]

    # Normalizing the dataset based on agent's scheme
    X_train_bp = X_train
    X_test_bp = X_test
    X_train_pcn = X_train
    X_test_pcn = X_test
    if normalize:
        # TODO: Normalize these according to new scheme (i.e. both X_train and X_test)
        X_train_bp = anns[0].normalize(X_train)
        X_test_bp = anns[0].normalize(X_test)
        X_train_pcn = pcns[0].normalize(X_train)
        X_test_pcn = pcns[0].normalize(X_test)

    # Setting meta-params
    epochs = settings.epochs
    sample_interval = np.ceil(epochs/10).astype(int)
    dataset_name = settings.dataset

    print(f'\nTraining {len(anns)} ANN agents and {len(pcns)} PCN agents')
    print(f'Number of epochs: {epochs}')
    print(f'Dataset: {dataset_name}')
    print()

    # Starting flop counter
    high.start_counters([events.PAPI_DP_OPS])

    # Training ANN agent
    print("\n ------------------ ANN Training ------------------ \n")
    anns_training_flops = []
    print("Training agents...")
    for agent in anns: 
        anns_training_flops.append(train_agent(agent, X_train_bp, y_train, epochs, verbose, sample_interval)[1])
        if verbose:
            print(f'finished training agent #{len(anns_training_flops)}')

    print(f'Average flops of agents over #{epochs} epochs: {np.mean(anns_training_flops):,}'.replace(',', ' '))
    print(f'Average Flops per epoch: {np.mean(anns_training_flops)/epochs:,}'.replace(',', ' '))
    print("\n ---------------------------------------------------- \n")

    # Training PCN agent
    print("\n ------------------ PCN Training ------------------ \n")
    pcns_training_flops = []
    print("Training agents...")
    for agent in pcns: 
        pcns_training_flops.append(train_agent(agent, X_train_pcn, y_train, epochs, verbose, sample_interval)[1])
        if verbose:
            print(f'finished training agent #{len(pcns_training_flops)}')

    print(f'Average flops of agents over #{epochs} epochs: {np.mean(pcns_training_flops):,}'.replace(',', ' '))
    print(f'Average Flops per epoch: {np.mean(pcns_training_flops)/epochs:,}'.replace(',', ' '))
    print("\n ---------------------------------------------------- \n")

    # Testing the ANN agent
    print('\n\n ------------------ ANN validation ------------------ \n')
    print('*** Training set ***')
    anns_accs_train = []
    anns_flops_train = []
    for agent in anns:
        high.read_counters()
        anns_accs_train.append(agent.test(X_train_bp, y_train, normalize_inputs=False)[1])
        anns_flops_train.append(high.read_counters()[0])
    print(f'Mean Accuracy: {np.mean(anns_accs_train)}')
    print(f'Flops performed: {np.mean(anns_flops_train):,}'.replace(',', ' '))
    print('*******************')
    
    print('*** Test set ***')
    anns_accs_test = []
    anns_flops_test = []
    for agent in anns:
        high.read_counters()
        anns_accs_test.append(agent.test(X_test_bp, y_test, normalize_inputs=False)[1])
        anns_flops_test.append(high.read_counters()[0])
    print(f'Accuracy: {np.mean(anns_accs_test)}')
    print(f'Flops performed: {np.mean(anns_flops_test):,}'.replace(',', ' '))
    print('*******************')

    print('\n ---------------------------------------------------- \n')

    # Testing the PCN agent
    print('\n\n ------------------ PCN validation ------------------ \n')
    print('*** Training set ***')
    pcns_accs_train = []
    pcns_flops_train = []
    for agent in pcns:
        high.read_counters()
        pcns_accs_train.append(agent.test(X_train_pcn, y_train, normalize_inputs=False)[1])
        pcns_flops_train.append(high.read_counters()[0])
    print(f'Accuracy: {np.mean(pcns_accs_train)}')
    print(f'Flops performed: {np.mean(pcns_flops_train):,}'.replace(',', ' '))
    print('*******************')

    print('*** Test set ***')
    pcns_accs_test = []
    pcns_flops_test = []
    for agent in pcns:
        high.read_counters()
        pcns_accs_test.append(agent.test(X_test_pcn, y_test, normalize_inputs=False)[1])
        pcns_flops_test.append(high.read_counters()[0])
    print(f'Accuracy: {np.mean(pcns_accs_test)}')
    print(f'Flops performed: {np.mean(pcns_flops_test):,}'.replace(',', ' '))
    print('*******************')

    print('\n ---------------------------------------------------- \n')

    # Saving results to csv file
    if save:
        df = pd.DataFrame({
            'agent' : range(len(anns)),
            'ann_tflops': anns_training_flops,
            'pcn_tflops': pcns_training_flops,
            'ann_train_acc': anns_accs_train,
            'ann_train_flops': anns_flops_train,
            'ann_test_acc': anns_accs_test,
            'ann_test_flops': anns_flops_test,
            'pcn_train_acc': pcns_accs_train,
            'pcn_train_flops': pcns_flops_train,
            'pcn_test_acc': pcns_accs_test,
            'pcn_test_flops': pcns_flops_test
        })
        df.to_csv('./experiments/flops/results/'+ dataset_name +'/epoch_stopping.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1:])