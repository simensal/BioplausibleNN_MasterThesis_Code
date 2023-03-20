import numpy as np
import pandas as pd
from pypapi import events, papi_high as high
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
    save = "-s" in argv or "--save" in argv

    # Get the data
    X_train, X_test, y_train, y_test = get_iris_data()

    # Initialize set of agents
    agents = [PCN(X_train.shape[1], [6], y_train.shape[1]) for _ in range(10)]

    # Normalizing the dataset based on agent's scheme
    X_train = agents[0].normalize(X_train)
    X_test = agents[0].normalize(X_test)

    # Setting meta-params
    epochs = 30
    sample_interval = np.ceil(epochs/10).astype(int)
    
    print(f'Number of agents: {len(agents)}')
    print(f'Number of epochs: {epochs}\n')

    # Starting flop counter
    high.start_counters([events.PAPI_DP_OPS])

    # Training the agent
    agent_flops = []
    print("Training agents...")
    for agent in agents: 
        agent_flops.append(train_agent(agent, X_train, y_train, epochs, verbose, sample_interval)[1])
        if verbose:
            print(f'finished training agent #{len(agent_flops)}')
    print("finished training agents.\n\n")

    # Average flops over n agents
    avg_flops = np.mean(agent_flops)
    print(f'Average flops performed during training: {avg_flops:,}'.replace(",", " "))

    # Testing the agents
    print('\n\n------------------ Validation set ------------------\n')
    agent_accs_test = []
    agent_flops_test = []
    for agent in agents:
        high.read_counters() # Resets counter for the agent
        agent_accs_test.append(agent.test(X_test, y_test, normalize_inputs=False)[1])
        agent_flops_test.append(high.read_counters()[0])
    print(f'Accuracy: {np.mean(agent_accs_test)}')
    print(f'Flops performed during test validation: {np.mean(agent_flops_test):,}'.replace(",", " "))
    print("\n---------------------------------------------------\n")

    # Printing the flops
    print('\n------------------ Training set ------------------\n')
    agent_accs_train = []
    agent_flops_train = []
    for agent in agents:
        high.read_counters() # Resets counter for the agent
        agent_accs_train.append(agent.test(X_train, y_train, normalize_inputs=False)[1])
        agent_flops_train.append(high.read_counters()[0])    
    print(f'Accuracy: {np.mean(agent_accs_train)}')
    print(f'Flops performed during training validation: {np.mean(agent_flops_train):,}'.replace(",", " "))
    print("\n---------------------------------------------------\n")
    high.stop_counters()

    # Save the results to pandas dataframe
    if save:
        df = pd.DataFrame({
            'agent': range(len(agents)), 
            'training_flops' : agent_flops,
            'validation_flops_train': agent_flops_train, 
            'validation_flops_test': agent_flops_test, 
            'validation_acc_train': agent_accs_train, 
            'validation_acc_test': agent_accs_test
            })
        df.to_csv('./experiments/PCN_multiagent_flops.csv', index=False)

if __name__ == "__main__":
    main(sys.argv[1:])