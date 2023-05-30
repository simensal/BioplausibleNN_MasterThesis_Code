# Imports
import numpy as np
import pandas as pd
import sys

from ANN import ANN
from PCN import PCN, PCN_decay
from helpers import mask_data, InferenceSettings, MAE, normalize, normalize_tanh, rescale_normalized_tanh
from sklearn.impute import KNNImputer

def main(argv):
    # Parsing arguments
    if '-d' not in argv:
        print('Error: no dataset provided.\nPlease specify a dataset to run the experiment on usind the -d flag followed by one of the following options: iris, mnist, wine')
        exit()

    verbose = '-v' in argv

    # Creating settings according to passed arguments
    settings = InferenceSettings(argv[argv.index('-d') + 1])

    # Load data from dataset, normalize, and mask it
    X_train_raw, X_test_raw, y_train, y_test = settings.get_data()

    if settings.dataset == 'mnist':
        X_train = 2 * X_train_raw / 256 - 1
        X_test = 2 * X_test_raw / 256 - 1
    else:
        X_train, X_test = normalize_tanh(X_train_raw, X_test_raw)

    masking_fractions = settings.masking_fractions
    sparse_datasets = []
    for fraction in masking_fractions:
        sparse_datasets.append((mask_data(X_train, fraction), mask_data(X_test, fraction)))

    # Train a pcn on the fully observed dataset
    if verbose:
        print('Training a PCN on the fully observed dataset...')
    pcn_fully = PCN_decay(X_train.shape[1], settings.hidden_layers, y_train.shape[1], activation_decay=settings.activation_decay, weight_decay=settings.weight_decay)

    for _ in range(settings.epochs):
        pcn_fully.train(X_train, y_train, normalize_inputs=False)

    # Infer the missing values of the sparse datasets using the fully trained pcn
    if verbose:
        print('Inferring the missing values of the sparse datasets...')

    pcn_reconstructed = []
    for dataset in sparse_datasets:
        pcn_reconstructed.append(pcn_fully.infer_missing_values(dataset[0], y_train, normalize_inputs=False))

    # Infering the missing 
    knn_reconstructed = []
    for dataset in sparse_datasets:
        knn = KNNImputer(n_neighbors=5)
        knn_reconstructed.append(knn.fit_transform(dataset[0]))

    pcn_reconstructed = np.array(pcn_reconstructed)
    knn_reconstructed = np.array(knn_reconstructed)

    # Rescaling the infered datasets to the size of the original dataset
    for i in range(len(pcn_reconstructed)):
        pcn_reconstructed[i] = rescale_normalized_tanh(pcn_reconstructed[i], X_train_raw)
        knn_reconstructed[i] = rescale_normalized_tanh(knn_reconstructed[i], X_train_raw)

    # Saving the reconstructed datasets
    if verbose:
        print('Saving the reconstructed datasets...')
    for i, dataset in enumerate(sparse_datasets):
        df = pd.DataFrame(np.isnan(dataset[0]))
        df.to_csv('./experiments/inference/results/'+ settings.dataset +'/'+str(settings.masking_fractions[i])+'/mask.csv')
    for i, dataset in enumerate(pcn_reconstructed):
        df = pd.DataFrame(dataset)
        df['label'] = np.argmax(y_train, axis=1)
        df.to_csv('./experiments/inference/results/'+ settings.dataset +'/'+str(settings.masking_fractions[i])+'/pcn.csv')
    for i, dataset in enumerate(knn_reconstructed):
        df = pd.DataFrame(dataset)
        df['label'] = np.argmax(y_train, axis=1)
        df.to_csv('./experiments/inference/results/'+ settings.dataset +'/'+str(settings.masking_fractions[i])+'/knn.csv')
    df = pd.DataFrame(X_train_raw)
    df['label'] = np.argmax(y_train, axis=1)
    df.to_csv('./experiments/inference/results/'+ settings.dataset +'/original.csv')


    ### MAE of the infered values ### 
    if verbose:
        print('Calculating the MAE of the inferred values, and saving the results...')
    pcn_loss = []
    knn_loss = []
    for i, dataset in enumerate(sparse_datasets):
        mask = np.isnan(dataset[0])
        df = pd.DataFrame({'pcn_infered': pcn_reconstructed[i][mask], 'knn_infered': knn_reconstructed[i][mask], 'real': X_train_raw[mask]})
        df.to_csv('./experiments/inference/results/'+ settings.dataset +'/'+str(settings.masking_fractions[i])+'inferred.csv')
        
        pcn_mae = MAE(X_train_raw[mask], pcn_reconstructed[i][mask])
        knn_mae = MAE(X_train_raw[mask], knn_reconstructed[i][mask])

        if verbose:
            print(f'MAE PCN at {settings.masking_fractions[i]} sparsity: {pcn_mae}')
            print(f'MAE KNN at {settings.masking_fractions[i]} sparsity: {knn_mae}\n')

        pcn_loss.append(pcn_mae)
        knn_loss.append(knn_mae)

    pd.DataFrame({'sparsity': masking_fractions,'pcn_loss': pcn_loss, 'knn_loss': knn_loss}).to_csv('./experiments/inference/results/'+ settings.dataset +'./loss.csv')
    
    ### MAE of the infered values ###


    ### Training agents on the reconstructed datasets ###

    # Train ten agents of each framework on the reconstructed datasets 

    if verbose:
        print('Training agents on the reconstructed datasets.')

    for i, dataset in enumerate(pcn_reconstructed):

        if verbose: 
            print(f'PCN reconstructed dataset #{i+1}')

        # Creating the agents
        anns = [ANN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]
        pcns = [PCN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]

        # Normalizing datasets using one of the agents
        if settings.dataset == 'mnist':
            X_train_ann = dataset / 256
            X_train_pcn = 2 * dataset / 256 - 1
            eval_test_ann = X_test_raw / 256
            eval_train_ann = X_train_raw / 256
        else:
            eval_train_ann, eval_test_ann = anns[0].normalize(X_train_raw, X_test_raw)
            X_train_ann, _ = anns[0].normalize(dataset, X_test)
            X_train_pcn, _ = pcns[0].normalize(dataset, X_test)

        # Training and testing the agents
        anns_acc = []
        pcns_acc = []
        anns_acc_train = []
        pcns_acc_train = []
        for ann, pcn in zip(anns, pcns):


            # Training the agent
            for _ in range(settings.epochs):
                pcn.train(X_train_pcn, y_train, normalize_inputs=False)
            for _ in range(settings.ann_eval_epochs):
                ann.train(X_train_ann, y_train, normalize_inputs=False)
            
            # Testing the agent on test split
            anns_acc.append(ann.test(eval_test_ann, y_test, normalize_inputs=False)[1])
            pcns_acc.append(pcn.test(X_test, y_test, normalize_inputs=False)[1])
            
            # Testing the agent on the real train split
            anns_acc_train.append(ann.test(eval_train_ann, y_train, normalize_inputs=False)[1])
            pcns_acc_train.append(pcn.test(X_train, y_train, normalize_inputs=False)[1])
            
            if verbose:
                print(f'Finished agents #{anns.index(ann)}')

        # Saving the results
        pd.DataFrame({'anns_acc': anns_acc, 'pcns_acc': pcns_acc}).to_csv('./experiments/inference/results/'+ settings.dataset +'/pcn_reconstructed/'+str(settings.masking_fractions[i])+'acc_test.csv')	
        pd.DataFrame({'anns_acc_train': anns_acc_train, 'pcns_acc_train': pcns_acc_train}).to_csv('./experiments/inference/results/'+ settings.dataset +'/pcn_reconstructed/'+str(settings.masking_fractions[i])+'acc_train.csv')
    
    for i, dataset in enumerate(knn_reconstructed):

        if verbose: 
            print(f'KNN reconstructed dataset #{i+1}')

        # Creating the agents
        anns = [ANN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]
        pcns = [PCN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]

        # Normalizing datasets using one of the agents
        if settings.dataset == 'mnist':
            X_train_pcn = 2 * dataset / 256 - 1
            X_train_ann = dataset / 256
            eval_test_ann = X_test_raw / 256
            eval_train_ann = X_train_raw / 256
        else: 
            eval_train_ann, eval_test_ann = anns[0].normalize(X_train_raw, X_test_raw)
            X_train_ann, _ = anns[0].normalize(dataset, X_test)
            X_train_pcn, _ = pcns[0].normalize(dataset, X_test)

        # Training and testing the agents
        anns_acc = []
        pcns_acc = []
        anns_acc_train = []
        pcns_acc_train = []
        for ann, pcn in zip(anns, pcns):
            # Training the agent
            for _ in range(settings.epochs):
                pcn.train(X_train_pcn, y_train, normalize_inputs=False)
            for _ in range(settings.ann_eval_epochs):
                ann.train(X_train_ann, y_train, normalize_inputs=False)

            # Testing the agent on test split
            anns_acc.append(ann.test(eval_test_ann, y_test, normalize_inputs=False)[1])
            pcns_acc.append(pcn.test(X_test, y_test, normalize_inputs=False)[1])
            
            # Testing the agent on the real train split
            anns_acc_train.append(ann.test(eval_train_ann, y_train, normalize_inputs=False)[1])
            pcns_acc_train.append(pcn.test(X_train, y_train, normalize_inputs=False)[1])

            if verbose:
                print(f'Finished agents #{anns.index(ann)}')

        # Saving the results
        pd.DataFrame({'anns_acc': anns_acc, 'pcns_acc': pcns_acc}).to_csv('./experiments/inference/results/'+ settings.dataset +'/knn_reconstructed/'+str(settings.masking_fractions[i])+'acc_test.csv')	
        pd.DataFrame({'anns_acc_train': anns_acc_train, 'pcns_acc_train': pcns_acc_train}).to_csv('./experiments/inference/results/'+ settings.dataset +'/knn_reconstructed/'+str(settings.masking_fractions[i])+'acc_train.csv')

    # Training control agents on the real datasets

    if verbose:
        print('Training control agents on the real datasets.')

    anns = [ANN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]
    pcns = [PCN(X_train.shape[1], settings.hidden_layers, y_train.shape[1]) for _ in range(10)]

    if settings.dataset == 'mnist':
        X_train_ann = X_train_raw / 256
        X_test_ann = X_test_raw / 256
    else:
        X_train_ann, X_test_ann = anns[0].normalize(X_train_raw, X_test_raw)
        # X_test_ann, _ = anns[0].normalize(X_test_raw, X_test_raw)

    anns_acc_control = []
    pcns_acc_control = []
    anns_acc_control_train = []
    pcns_acc_control_train = []

    for ann, pcn in zip(anns, pcns): 
        for _ in range(settings.epochs):
            pcn.train(X_train, y_train, normalize_inputs=False)
        for _ in range(settings.ann_eval_epochs):
            ann.train(X_train_ann, y_train, normalize_inputs=False)
        
        anns_acc_control.append(ann.test(X_test_ann, y_test, normalize_inputs=False)[1])
        pcns_acc_control.append(pcn.test(X_test, y_test, normalize_inputs=False)[1])

        anns_acc_control_train.append(ann.test(X_train_ann, y_train, normalize_inputs=False)[1])
        pcns_acc_control_train.append(pcn.test(X_train, y_train, normalize_inputs=False)[1])
    
    pd.DataFrame({'anns_acc_control': anns_acc_control, 'pcns_acc_control': pcns_acc_control, 'anns_acc_control_train': anns_acc_control_train, 'pcns_acc_control_train': pcns_acc_control_train}).to_csv('./experiments/inference/results/'+ settings.dataset +'/control_acc_test.csv')

    ### Training agents on the reconstructed datasets ###

if __name__ == '__main__':
    main(sys.argv[1:])