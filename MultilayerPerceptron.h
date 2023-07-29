#pragma once

#include <cstdlib>
#include <vector>
#include "ActivationFunction.h"
#include "Layer.h"
#include "data/Dataset.h"

using namespace std;

class MultilayerPerceptron
{
public:
	explicit MultilayerPerceptron(const vector<int>& shape, ActivationFunction& activationFunction);

	void Train(Dataset dataset, double learningRate, int epochs);

	vector<double> ForwardPass(vector<double>);
private:
	int NumberOfInputs();
	int NumberOfOutputs();

	vector<int> shape;

	vector<Layer> layers;
};
