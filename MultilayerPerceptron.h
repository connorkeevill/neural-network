#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>
#include "ActivationFunction.h"
#include "Layer.h"
#include "Dataset.h"
#include "CostFunction.h"
#include "ThreadPool/ThreadPool.h"

using namespace std;

class MultilayerPerceptron
{
public:
	MultilayerPerceptron(const vector<int>& shape, ActivationFunction& activationFunction, CostFunction& costFunction);

	void Train(Dataset* dataset, double learningRate, int batchSize, int epochs);

	vector<double> ForwardPass(vector<double>);
private:
	vector<vector<double>> TrainingForwardPass(vector<double> input);

	int NumberOfInputs();
	int NumberOfOutputs();

	CostFunction& costFunction;

	vector<int> shape;
	vector<Layer> layers;
};

struct BackpropData {
	vector<vector<double>> activations;
};