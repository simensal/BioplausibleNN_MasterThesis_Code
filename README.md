# Bio-plausible Neural Networks - A Comparative Study of the Computational Demand and Capability to Infer Missing Values of Predictive Coding Networks

This repository contains the source code developed for the research of a master thesis written at [Norwegian University for Science and Technology](https://www.ntnu.no/). The thesis explores the computational demand of predictive coding networks (PCN) and their capability to infer missing values in incomplete datasets. The repository is meant as supplementary material for readers of the thesis interested in exploring the experiment results and/or further research using the system. Comments in the code are not meant as documentation and is only used for personal understanding. 

This README will provide a short overview of the repository. It consists of three key parts; the code implementing the frameworks, the experiments run for the thesis using the frameworks implemented by the code, and the resulting data from the experiments. 

## Framework code
All files related to the frameworks are included in the root folder of the repository, and their function are summarized in the table below. 

| **file**     | **function**                                                                                                                                           |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `learner.py` | Learner interface for the ANN and PCN frameworks. \n Declares the ´train´, ´test´, and ´predict´ methods                                               |
| `ANN.py`     | Framework for setting up artificial neural networks to solve classification tasks.                                                                     |
| `PCN.py`     | Framework for setting up predictive coding network models to solve classification tasks.                                                               |
| `helpers.py` | Misc pieces of code which have been reused repeatedly. \nIncludes mathematical functions, data loading and settings for experiments among other things |

## Scripts
The scripts uses the frameworks to perform the experiments described in the thesis. For this, there are four files, three of which belongs to the computational analysis (in this repository referred to as flop), and the forth belonging to the inference experiment. The table below described which part of the experiment each file is responsible for

| **file**             	| **function**                                                                                                                      	|
|----------------------	|-----------------------------------------------------------------------------------------------------------------------------------	|
| `inference_exp.py`   	| Performs the inference using KNN and PCN on incomplete datasets, and the post-training evaluation of the agents of each framework 	|
| `flop_exp__epoch.py` 	| Performs the equal epoch training approach for the agents in the computational analysis                                           	|
| `flop_exp__flops.py` 	| Performs the equal flops training approach for the agents in the computational analysis                                           	|
| `flop_exp__acc.py`   	| Performs the equal accuracy training approach for the agents in the computational analysis                                        	|

## Results
Lastly, the `/experiments` folder contains all the data collected from running the experiments. The folder is further divided into two subfolder, one for the inference experiment and another for the flop experiment. In both cases, there will be a subfolder under the subfolder `/results` with the data for each dataset the experiment was run on. The next two segments will explain the individual structures found within these subfolders. 

### Inference
Dataset-folders for the inference experiment contain the original complete dataset in `original.csv`. Further, there is a folder for each sparsity level with the mask of masked values in `mask.csv` and the reconstructed datasets for the PCN and KNN approach in `pcn.csv` and `knn.csv` respectively. The accuracies of the controll agents trained on the complete dataset without any values missing are contain in `control_acc_test.csv`. Lastly, the folders named `pcn_reconstructed` and `knn_reconstructed` contain the accuracies recorded for the agents trained on the reconstructed datasets for the $D_{PCN}$ and $D_{KNN}$, respectively. The files within these folders are prefixed with the sparsity degree of the reconstructed dataset the agents were trained on, and suffixed by which split the accuracy is collected from.  

### Flops
For the flop experiment, the results are stored in two files and a folder. `epoch_stopping.csv` contains the data from equal epoch, `flop_stopping.csv` contains the data from the equal flops approach, and the `acc_stopping` folder contains the data in `main.csv` - supplemented by the training accuracies and flops of the ANN and PCN agents in `anns.csv` and `pcns.csv`, respectively.  

## Misc
The dataset used are made available in the `/data` folder, and lastly, all required 3rd party libs used in the code can be installed by running 

<!-- The following repository contains the code for two frameworks. One to create neural networks using back propagation, and the second for predictive coding networks. The repository serves as the main hub for code and experiments related to a master thesis done at the . 

Lastly, all necessary 3rd-party libs are available in [`requirements.txt`](./requirements.txt) and can be installed using  -->

```sh
pip install -r ./requirements.txt
```
