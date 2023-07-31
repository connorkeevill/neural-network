#pragma once

#include <vector>
#include <shared_mutex>
#include "ActivationFunction.h"

using namespace std;

class Neuron {
public:
	Neuron(int numberOfInputs, ActivationFunction& activationFunction);

	double ForwardPass(vector<double> input);
	double GetWeight(int index);
	double UpdateGradients(vector<double> &previousActivations, double activation, double nextLayerPartialDerivative);
	void ApplyGradientsToWeights(double scalingFactor);
private:
	int numberOfInputs;
	ActivationFunction& activationFunction;

	mutex *weightGradientMutex;
	mutex *biasGradientMutex;

	vector<double> weights;
	vector<double> weightGradients;
	double bias;
	double biasGradient;
};
