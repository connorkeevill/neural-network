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
	auto network = std::make_unique<MultilayerPerceptron>(std::vector<int>{784, 3, 10}, activationFunction, costFunction);

	Dataset data = MnistDataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

	for(int i = 0; i < 5; ++i)
	{
		FeatureVector fv = data.GetNextFeatureVector();
		vector<double> out = network->ForwardPass(fv.data);

		for(double val : out)
		{
			cout << val << " ";
		}

		cout << endl;

		for(double val : fv.label)
		{
			cout << val << " ";
		}

		cout << endl;
		cout << endl;
	}
}
