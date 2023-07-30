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

	Dataset trainingData = MnistDataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");

	network->Train(trainingData, 0.1, 100);

	Dataset testData = MnistDataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

	for(int i = 0; i < 10; ++i)
	{
		FeatureVector fv = testData.GetNextFeatureVector();

		vector<double> predicted = network->ForwardPass(fv.data);

		for(double element : predicted)
		{
			cout << element << " ";
		}

		cout << endl;

		for(double element : fv.label)
		{
			cout << element << " ";
		}

		cout << endl;
	}
}
