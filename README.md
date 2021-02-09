# PINE: Parallel Interpreter Network

## Introduction
PINE (**P**arallel **I**nterpreter **NE**twork) is a novel interpretability framework which provides decent interpretations of DNNs in order to make the reasoning behind decisions of a black-box model become transparent to users.

## How PINE works?
PINE's structure consists of two paralell networks. The Main Model, which is the primary model we want to get interpreted, and the Interperter, which is an autoencoder network trains parallel to the main model and eventualy learns how the main model predicts. 
![image](https://user-images.githubusercontent.com/19486359/107159400-a3878080-6987-11eb-9075-bef8251559a4.png)

## Experiments
After experimenting PINE on an MNIST model, the results below have achieved:
# MNIST
![image](https://user-images.githubusercontent.com/19486359/107133665-a11f1b00-68e2-11eb-99ed-33839a32c844.png)

## TODO
- [ ] Add PINE visualizations
- [ ] Add comparison with other Interpretability Methods
- [ ] Add PINE results on Cifar10 
