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
	auto network = std::make_unique<MultilayerPerceptron>(std::vector<int>{784, 10}, activationFunction, costFunction);

	Dataset *trainingData = new MnistDataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");

	network->Train(trainingData, 0.2, 32, 20);

	Dataset *testData = new MnistDataset("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte");

	int correctClassifications = 0;

	while(!testData->EndOfData())
	{
		FeatureVector fv = testData->GetNextFeatureVector();
		vector<double> predicted = network->ForwardPass(fv.data);

		string predictedClass = testData->ClassificationToString(predicted);
		string actualClass = testData->ClassificationToString(fv.label);

		if(predictedClass == actualClass) { ++correctClassifications; }

		cout << "Predicted: " << predictedClass << endl;
		cout << "Actual: " << actualClass << endl;
		cout << endl;
	}

	cout << "Correct classifications: " << correctClassifications << endl;
	cout << "Accuracy: " << (correctClassifications / 10000.0) << endl;
}
