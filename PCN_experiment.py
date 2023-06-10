from PCN import PCN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plyer import notification
from helpers import get_iris_data

epochs = 300
sample_interval = np.ceil(epochs / 15).astype(int)

if __name__ == "__main__":
    # Load data from iris and create training and test sets
    X_train, X_test, y_train, y_test = get_iris_data()

    # Setting up PCN agent
    agent = PCN(4, [6] ,3, learning_rate=0.05)

    # Training the agent
    print('Training the agent...')
    agent_checkpoints = []
    for i in range(epochs):
        agent.train(X_train, y_train)
        if not i % sample_interval:
            print('Epoch: ', i)
            pred, accuracy, err = agent.test(X_test, y_test)
            agent_checkpoints.append(np.mean(accuracy))
    
    # Plotting the average accuracy of the agents
    plt.plot(range(0, epochs, sample_interval), agent_checkpoints)
    plt.title('PCN accuracy per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # Testing the agent
    print('Testing the agent...')
    pred, accuracy, err = agent.test(X_test, y_test)
    print('Predictions: ', pred)
    print('Accuracy: ', accuracy)
    print('Error: ', err)


    # Notification
    notification.notify(
        title='Experiment Complete',
        message='The experiment has completed. Please check the graphs.',
        app_name='PCN',
        timeout=10
    )
