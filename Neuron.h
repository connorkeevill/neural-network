#pragma once

#include <vector>
#include "ActivationFunction.h"

using namespace std;

class Neuron {
public:
	Neuron(int numberOfInputs, ActivationFunction& activationFunction);

	double ForwardPass(vector<double> input);

	double GetActivation();
private:
	int numberOfInputs;
	ActivationFunction& activationFunction;

	vector<double> weights;
	double bias;

	double activation;
};
