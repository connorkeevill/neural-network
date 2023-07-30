#include <iostream>
#include "data/Dataset.h"
#include "MultilayerPerceptron.h"
#include "ActivationFunction.h"
#include "CostFunction.h"

int main()
{
	// Create a network with 3 input neurons, one hidden layer with 3 neurons, and an output layer of 2 neurons.
	Sigmoid activationFunction{};
	MeanSquaredError costFunction{};
	auto network = std::make_unique<MultilayerPerceptron>(std::vector<int>{784, 100, 10}, activationFunction, costFunction);

	Dataset data = MnistDataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

	network->Train(data, 0.5, 1000);
}
