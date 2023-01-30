# Bioplausible neural networks 🚧WIP🚧

The following repository contains the code for two frameworks. One to create neural networks using back propagation, and the second for predictive coding networks. The repository serves as the main hub for code and experiments related to a master thesis done at the [Norwegian University for Science and Technology](https://www.ntnu.no/). 

The two frameworks are located in the files [`ANN.py`](./ANN.py) and [`PCN.py`](./PCN.py) respectively, and they both build on the common framework found in [`learner.py`](./learner.py). Secondly, the frameworks have been used in a series of experiments, where they are applied to the iris dataset from UCI and the MNIST dataset. The related code can be found in the folder [`/experiments`](./experiments).

Lastly, all necessary 3rd-party libs are available in [`requirements.txt`](./requirements.txt) and can be installed using 

```sh
pip install -r ./requirements.txt
```
