#pragma once

#include <cstdlib>
#include <vector>
#include <functional>
#include "Layer.h"
#include "data/Dataset.h"

using namespace std;

class MultilayerPerceptron
{
public:
	explicit MultilayerPerceptron(const vector<int>& shape, const function<double(double)> activationFunction);

	void Train(const Dataset dataset);

	vector<double> ForwardPass(vector<double>);
private:
	int NumberOfInputs();
	int NumberOfOutputs();

	vector<int> shape;

	vector<Layer> layers;
};
