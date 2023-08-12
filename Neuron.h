#pragma once

#include <vector>
#include <shared_mutex>
#include "ActivationFunction.h"

using namespace std;

class Neuron {
public:
	Neuron(int numberOfInputs);

	double ForwardPass(vector<double> input);
	double GetWeight(int index);
	double UpdateGradients(vector<double> &previousActivations, double activationDerivative, double nextLayerPartialDerivative);
	void ApplyGradientsToWeights(double scalingFactor);
private:
	int numberOfInputs;

	mutex *gradientMutex;

	vector<double> weights;
	vector<double> weightGradients;
	double bias;
	double biasGradient;
};
