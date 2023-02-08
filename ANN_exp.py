from ANN import ANN
import numpy as np
import matplotlib.pyplot as plt
from plyer import notification
from helpers import get_iris_data

epochs = 100
sample_interval = np.ceil(epochs / 15).astype(int)

if __name__ == "__main__":

    # Load data from iris and create training and test sets
    X_train, X_test, y_train, y_test = get_iris_data()

    # Setting up ten agents of ANN
    agents = [ANN(4, [6], 3, learning_rate=0.05) for _ in range(10)]

    # Training the agents
    print('Training the agents...')
    agent_accuracies = []
    agent_errors = []
    for agent in agents:
        accuracies = []
        errors = []
        for i in range(epochs):
            if not i % sample_interval:
                pred, acc, err = agent.test(X_test, y_test)
                accuracies.append(np.mean(acc))
                errors.append(np.mean(err))
            agent.train(X_train, y_train)
        agent_accuracies.append(accuracies)
        agent_errors.append(errors)

    # Calculating the average error of the agents
    avg_accuracy = np.mean(agent_accuracies, axis=0)
    avg_errors = np.mean(agent_errors, axis=0)

    # Notification
    notification.notify(
        title='Experiment Complete',
        message='The experiment has completed. Please check the graphs.',
        app_name='ANN',
        timeout=10
    )

    # Plotting the average error of the agents
    plt.plot(range(0, epochs, sample_interval), avg_errors)
    plt.xlabel('Epochs')
    plt.ylabel('Average Error')
    plt.title('Average Error of Agents')
    plt.show()

    for i in range(len(agents)):
        plt.plot(range(0, epochs, sample_interval),
                 agent_errors[i], label='Agent ' + str(i))
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error of Agents')
    plt.legend()
    plt.show()

    test_predictions = []
    test_accuracies = []
    test_errors = []

    train_predictions = []
    train_accuracies = []
    train_errors = []

    # Testing the agents
    for agent in agents:
        pred, acc, err = agent.test(X_test, y_test)
        test_predictions.append(pred)
        test_accuracies.append(acc)
        test_errors.append(err)

        pred, acc, err = agent.test(X_train, y_train)
        train_predictions.append(pred)
        train_accuracies.append(acc)
        train_errors.append(err)

    # Finding the prediction accuracies of the agents
    print('Test Accuracies: ', test_accuracies)
    print('Train Accuracies: ', train_accuracies)

    # Average accuracy of the agents
    print('Average Accuracy: ', np.mean(test_accuracies))
    print('Average Train Accuracy: ', np.mean(train_accuracies))

    # Median accuracy of the agents
    print('Median Accuracy: ', np.median(test_accuracies))
    print('Median Train Accuracy: ', np.median(train_accuracies))
