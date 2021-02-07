# PINE: Parallel Interpreter Network

## Introduction
PINE (**P**arallel **I**nterpreter **NE**twork) is a novel interpretability framework which provides a decent interpretations of DNNs in order to make the reasoning behind the decisions of a black-box model transparent to usrs.

## How PINE works?
PINE's structure consists of two paralell networks. The Main Model, whcih is the primary model we want to get interpreted, and the Interperter, which is an autoencoder network trains parallel to the main model and eventualy learns how the main model predicts. 

## Experiments

# MNIST
![image](https://user-images.githubusercontent.com/19486359/107133665-a11f1b00-68e2-11eb-99ed-33839a32c844.png)

## TODO
- [ ] Add PINE visualizations
- [ ] Add comparison with other Interpretability Methods
- [ ] Add PINE results on Cifar10 
