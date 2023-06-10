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
        # if verbose and (i+1) % sample_interval == 0:
        #     print(f'Finished epoch #{i+1}')

    return agent, high.read_counters()[0]

def test_agent(agent, X_test, y_test):
    """
    Test the agent on the given test set and return the accuracy and the number
    of flops performed during testing.

    Parameters
    ----------
    agent : ANN or PCN
        The agent to be tested.
    X_test : np.ndarray
        The test set.
    y_test : np.ndarray
        The test labels.

    Returns
    -------
    acc : float
        The accuracy of the agent on the test set.
    flops : int
        The number of flops performed during testing.
    """
    high.read_counters() # Resets counter for the agent
    _, acc, _ = agent.test(X_test, y_test, normalize_inputs=False)
    flops = high.read_counters()[0]

    return acc, flops

def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    save = "-s" in argv or "--save" in argv
    normalize = "-n" in argv or "--normalize" in argv

    n_agents = 10
    if '-n_agents' in argv:
        n_agents = int(argv[argv.index('-n_agents') + 1])

    max_epochs = 500
    if '-max_epochs' in argv:
        max_epochs = int(argv[argv.index('-max_epochs') + 1])
    

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
        X_train_bp = anns[0].normalize(X_train)
        X_test_bp = anns[0].normalize(X_test)
        X_train_pcn = pcns[0].normalize(X_train)
        X_test_pcn = pcns[0].normalize(X_test)


    # Setting meta-params
    # epochs = settings.epochs
    epochs = 10
    sample_interval = np.ceil(epochs/10).astype(int)
    dataset_name = settings.dataset
    stopping_acc = settings.stopping_acc

    print(f'\nTraining {len(anns)} ANN agents and {len(pcns)} PCN agents')
    print(f'Number of epochs: {epochs}')
    print(f'Dataset: {dataset_name}')
    print()

    # Starting flop counter
    high.start_counters([events.PAPI_DP_OPS])

    # Training ANN agent
    print("\n ------------------ ANN Training ------------------ \n")
    anns_training_flops = np.zeros(len(anns)) # 1d-array with the flops of each agent
    anns_testing_flops = [] # 1d-array with the flops of each agent
    anns_accs = [[0 for _ in range(n_agents)]] # 2d-array with the accuracy of each agent at sampling interval
    ann_training_epochs = 0 # Number of epochs trained
    print("Training agents...")
    while np.mean(anns_accs[-1]) < stopping_acc and ann_training_epochs < max_epochs:
        temp_train_flops = []
        temp_test_flops = []
        temp_accs = []
        for agent in anns: 
            temp_train_flops.append(train_agent(agent, X_train_bp, y_train, epochs, verbose, sample_interval)[1])
            acc, flops = test_agent(agent, X_test_bp, y_test)
            temp_accs.append(acc)
            temp_test_flops.append(flops)

        anns_training_flops += temp_train_flops
        anns_testing_flops.append(temp_test_flops)
        anns_accs.append(temp_accs)
        ann_training_epochs += epochs
        if verbose: 
            print(f'Finished epoch #{ann_training_epochs}')

    anns_accs = anns_accs[1:]

    print(f'Average flops of agents over #{ann_training_epochs} epochs: {np.mean(anns_training_flops):,}'.replace(',', ' '))
    print(f'Average Flops per epoch: {np.mean(anns_training_flops)/ann_training_epochs:,}'.replace(',', ' '))
    print("\n ---------------------------------------------------- \n")

    # Training PCN agent
    print("\n ------------------ PCN Training ------------------ \n")
    pcns_training_flops = np.zeros(len(pcns))
    pcns_testing_flops = []
    pcns_accs = [[0 for _ in range(n_agents)]]
    pcn_training_epochs = 0
    print("Training agents...")
    while np.mean(pcns_accs[-1]) < stopping_acc and pcn_training_epochs < max_epochs:
        temp_train_flops = []
        temp_test_flops = []
        temp_accs = []
        for agent in pcns: 
            temp_train_flops.append(train_agent(agent, X_train_pcn, y_train, epochs, verbose, sample_interval)[1])
            acc, flops = test_agent(agent, X_test_pcn, y_test)
            temp_accs.append(acc)
            temp_test_flops.append(flops)

        pcns_training_flops += temp_train_flops
        pcns_testing_flops.append(temp_test_flops)
        pcns_accs.append(temp_accs)
        pcn_training_epochs += epochs
        if verbose: 
            print(f'Finished epoch #{pcn_training_epochs}')

    pcns_accs = pcns_accs[1:]

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
            'ann_epochs': np.ones(len(anns))*ann_training_epochs,
            'pcn_tflops': pcns_training_flops,
            'pcn_epochs': np.ones(len(pcns))*pcn_training_epochs,
            'ann_train_acc': anns_accs_train,
            'ann_train_flops': anns_flops_train,
            'ann_test_acc': anns_accs_test,
            'ann_test_flops': anns_flops_test,
            'pcn_train_acc': pcns_accs_train,
            'pcn_train_flops': pcns_flops_train,
            'pcn_test_acc': pcns_accs_test,
            'pcn_test_flops': pcns_flops_test
        })
        df.to_csv('./experiments/flops/results/'+ dataset_name +'/acc_stopping/main.csv', index=False)
        
        df2 = pd.DataFrame({
            'ann_acc' + str(i) : np.array(anns_accs).T[i] for i in range(n_agents)
        })

        # df2 = df2.append({'ann_flop'+str(i) : np.array(anns_testing_flops).T[i] for i in range(n_agents)}, ignore_index=True)
        for i in range(n_agents):
            df2['ann_flop'+str(i)] = np.array(anns_testing_flops).T[i]
        df2.to_csv('./experiments/flops/results/'+ dataset_name +'/acc_stopping/anns.csv', index=False)


        df3 = pd.DataFrame({
            'pcn_acc' + str(i) : np.array(pcns_accs).T[i] for i in range(n_agents)
        })
        # df3 = df3.append({'pcn_flop'+str(i) : np.array(pcns_testing_flops).T[i] for i in range(n_agents)}, ignore_index=True)
        for i in range(n_agents):
            df3['pcn_flop'+str(i)] = np.array(pcns_testing_flops).T[i]
        df3.to_csv('./experiments/flops/results/'+ dataset_name +'/acc_stopping/pcns.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1:])