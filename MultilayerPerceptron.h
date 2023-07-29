#pragma once

#include <cstdlib>
#include <vector>
#include "ActivationFunction.h"
#include "Layer.h"
#include "data/Dataset.h"
#include "CostFunction.h"

using namespace std;

class MultilayerPerceptron
{
public:
	MultilayerPerceptron(const vector<int>& shape, ActivationFunction& activationFunction, CostFunction& costFunction);

	void Train(Dataset dataset, double learningRate, int epochs);

	vector<double> ForwardPass(vector<double>);
private:
	int NumberOfInputs();
	int NumberOfOutputs();

	CostFunction& costFunction;

	vector<int> shape;
	vector<Layer> layers;
};
