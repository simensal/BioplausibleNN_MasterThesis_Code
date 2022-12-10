from ANN import ANN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plyer import notification

epochs = 1000
sample_interval = 50

if __name__ == "__main__":

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


    # Setting up ten agents of ANN
    agents = [ANN(4, [6], 3, learning_rate=0.05) for _ in range(10)]


    # Training the agents
    print('Training the agents...')
    agent_checkpoints = []
    for agent in agents:
        checkpoints = []
        for i in range(epochs):
            if not i % sample_interval:
                error, pred = agent.test(X_test, y_test)
                checkpoints.append(np.mean(error))
            agent.train(X_train, y_train)
        agent_checkpoints.append(checkpoints)

    # Calculating the average error of the agents
    avg_errors = np.mean(agent_checkpoints, axis=0)

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
        plt.plot(range(0, epochs, sample_interval), agent_checkpoints[i], label='Agent ' + str(i))
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Error of Agents')
    plt.legend()
    plt.show()

    test_errors = []
    test_predictions = []
    train_errors = []
    train_predictions = []

    # TODO: Test the agents on the test set

    # Testing the agents
    for agent in agents:
        error, pred = agent.test(X_test, y_test)
        test_errors.append(error)
        test_predictions.append(pred)

        error, pred = agent.test(X_train, y_train)
        train_errors.append(error)
        train_predictions.append(pred)


    # Converting one hot encoding to labels
    y_test_labels = np.argmax(y_test, axis=1)
    y_train_labels = np.argmax(y_train, axis=1)
    # print('y-test labels: ', y_test_labels)

    # Finding the prediction accuracies of the agents
    test_accuracies = [np.sum(pred == y_test_labels)/len(y_test_labels) for pred in test_predictions]
    train_accuracies = [np.sum(pred == y_train_labels)/len(y_train_labels) for pred in train_predictions]
    print('Test Accuracies: ', test_accuracies)
    print('Train Accuracies: ', train_accuracies)

    # Average accuracy of the agents
    avg_test_accuracy = np.mean(test_accuracies)
    avg_train_accuracy = np.mean(train_accuracies)
    print('Average Accuracy: ', avg_test_accuracy)
    print('Average Train Accuracy: ', avg_train_accuracy)


    # Median accuracy of the agents
    median_test_accuracy = np.median(test_accuracies)
    median_train_accuracy = np.median(train_accuracies)
    print('Median Accuracy: ', median_test_accuracy)
    print('Median Train Accuracy: ', median_train_accuracy)

    print(agents[0].predict(X_test[0]))