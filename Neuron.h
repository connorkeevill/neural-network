#pragma once

#include <vector>
#include "ActivationFunction.h"

using namespace std;

class Neuron {
public:
	Neuron(int numberOfInputs, ActivationFunction& activationFunction);

	double ForwardPass(vector<double> input);
	double GetWeight(int index);
	void UpdateGradients(vector<double> previousActivations, double partialDerivative);
	void ApplyGradientsToWeights(double scalingFactor);

	double PartialDerivative;
private:
	int numberOfInputs;
	ActivationFunction& activationFunction;

	vector<double> weights;
	vector<double> weightGradients;
	double bias;
	double biasGradient;
};
