#pragma once

#include "Neuron.h"
#include <vector>
#include "ActivationFunction.h"

class Layer
{
public:
	Layer(int numberOfInputs, int numberOfOutputs, ActivationFunction& activationFunction);

	vector<double> ForwardPass(vector<double> inputs);
private:
	int numberOfInputs;
	int numberOfNeurons;

	vector<Neuron> neurons;
};
