#include <iostream>
#include "Dataset.h"
#include "MultilayerPerceptron.h"
#include "ActivationFunction.h"
#include "CostFunction.h"
#include "stopwatch/Stopwatch.hpp"

int main()
{
	Stopwatch stopwatch{};

	// Create a network with 3 input neurons, one hidden layer with 3 neurons, and an output layer of 2 neurons.
	Sigmoid hiddenLayerActivationFunction{};
	Softmax outputLayerActivationFunction{};
	MeanSquaredError costFunction{};
	auto network = std::make_unique<MultilayerPerceptron>(
			std::vector<int>{784, 100, 10},
			hiddenLayerActivationFunction,
			outputLayerActivationFunction,
			costFunction);

	stopwatch.addMeasurement("Before reading data.");
	Dataset *trainingData = new MnistDataset("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
	stopwatch.addMeasurement("Data read");


	network->Train(trainingData, 1, 100, 10);
	stopwatch.addMeasurement("Network trained");

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
		for(double element : predicted) { cout << element << " ";}
		cout << endl;
		cout << "Actual: " << actualClass << endl;
		for(double element : fv.label) { cout << element << " ";}
		cout << endl;
	}

	stopwatch.addMeasurement("Accuracy test completed");

	cout << "Correct classifications: " << correctClassifications << endl;
	cout << "Accuracy: " << (correctClassifications / 10000.0) << endl;

	cout << stopwatch.getTimingTrace();
}
