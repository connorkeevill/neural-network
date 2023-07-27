#pragma once

#include "Neuron.h"
#include <vector>
#include <functional>

class Layer
{
public:
	Layer(int numberOfInputs, int numberOfOutputs, const function<double(double)>& activationFunction);

	vector<double> ForwardPass(vector<double> inputs);
private:
	int numberOfInputs;
	int numberOfNeurons;

	vector<Neuron> neurons;
};
