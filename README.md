## Neural Network in C++

In this project I implement a basic neural network and backpropagation, to learn how neural networks work.

### Performance
I evaluated the network on the MNIST dataset, finding that it is able to achieve ~90% on a three layer network - 784 input neurons, 100 hidden neurons, 10 output neurons - using Sigmoid for the hidden layer activation function, and Softmax for the output layer activation, training with a batch size of 100 for 10 epochs at a learning rate of 0.5.

### Bugs
There are some outstanding bugs:
- Sometimes during training, the cost function can become nan. I think this is due to some sort of vanishing / exploding gradients problem, but I am yet to perform any numerical gradient checking and the stochastic nature of this bug (i.e., it seems to be sensitive to the random initial state of the network) makes it quite tricky to debug!
- The network does not train well with ReLU. I have a hunch that this is related to the way that the derivative is being calculted (maybe not!), but I am also yet to invesitgate.

Please feel free to open a pull request if you spot the cause of either of these bugs :)
