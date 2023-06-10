import numpy as np
from helpers import get_data, mask_data
from PCN import PCN, PCN_decay as PCN_decay
np.set_printoptions(precision = 3, suppress=True)
# Load data from iris and create training and test sets
X_train, X_test, y_train, y_test = get_data('./data/wine/wine.data')

# Setting a decay PCN
pcns = [PCN_decay(X_train.shape[1], [13], y_train.shape[1], activation_decay=5e-4, weight_decay=1e-4) for _ in range(10)]
plain_pcns = [PCN(X_train.shape[1], [13], y_train.shape[1]) for _ in range(10)]

# Normalizing the dataset using one of the agents
X_train = pcns[0].normalize(X_train)
X_test = pcns[0].normalize(X_test)

missing_datasets = []
fractions = [0.02*i for i in range(1, 5)]
for percentage in fractions:
    train = mask_data(X_train, percentage)
    test = mask_data(X_test, percentage)
    missing_datasets.append((train, test))

# Benchmark on missing datasets
loss_benchmark = []
for dataset in missing_datasets:
    sparce_dataset = np.nan_to_num(dataset[0])
    mask = np.isnan(dataset[0])
    loss_benchmark.append(np.mean(np.abs(sparce_dataset[mask] - X_train[mask])))
#print(f'Benchmark losses: \n{np.array(loss_benchmark)}\n')

# Training and testing the agents
plain_accs = []
for pcn in plain_pcns:
    for _ in range(10):
        pcn.train(X_train, y_train, normalize_inputs=False)
    _, acc, _ = pcn.test(X_test, y_test, normalize_inputs=False)
    plain_accs.append(acc)
accs = []
for pcn in pcns: 
    # print(f'Training agent #{pcns.index(pcn)}')
    for i in range(10): 
        pcn.train(X_train, y_train, normalize_inputs=False)
    pred, acc, err = pcn.test(X_test, y_test, normalize_inputs=False)
    accs.append(acc)
print(f'Last example in the train set: \n{y_test[-1]}\n')
print(f'Mean prediction accruacy:\n{np.mean(accs, axis=0):.3f}\n')
print(f'Mean benchmark accuracy:\n{np.mean(plain_accs, axis=0):.3f}\n')
print(f'Prediction head: \n{np.array(pred[:3] + pred[-3:])}\n')

# Training the agent on the missing datasets
pcns_2 = [PCN_decay(X_train.shape[1], [13], y_train.shape[1]) for _ in range(len(fractions))]
reconstructed = []
for dataset, agent in zip(missing_datasets, pcns_2):
    Z_train, Z_test = dataset
    for i in range(30):
        agent.train(Z_train, y_train, normalize_inputs=False)
    reconstructed.append(agent.infer_missing_values(Z_train, y_train, normalize_inputs=False))

# Printing the benchmark mean loss for comparison
print(f'Reconstruction MSE benchmark before any inference:\n{np.array(loss_benchmark)}\n')


# Calculating the mean loss of the predicted datasets
losses = np.array([])
for i, dataset in enumerate(reconstructed):
    mask = np.isnan(missing_datasets[i][0])
    loss = np.mean(np.abs(dataset[mask] - X_train[mask]))
    losses = np.append(losses, loss)
print(f'Reconstruction MSE for sparsely trained agents:\n{np.array(losses)}\n{loss_benchmark - losses}')

