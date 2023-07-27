#include <iostream>
#include "data/Dataset.h"
#include "MultilayerPerceptron.h"

int main()
{
	// Create a network with 3 input neurons, one hidden layer with 3 neurons, and an output layer of 2 neurons.
	auto network = std::make_unique<MultilayerPerceptron>(std::vector<int>{3, 3, 2}, [](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    });
	auto trainingData = std::make_unique<Dataset>();

	network->Train(*trainingData);
}
