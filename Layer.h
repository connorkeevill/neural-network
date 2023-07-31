#pragma once

#include "Neuron.h"
#include <vector>
#include "ActivationFunction.h"
#include "CostFunction.h"

class Layer
{
public:
	Layer(int numberOfInputs, int numberOfOutputs, ActivationFunction& activationFunction);

	vector<double> ForwardPass(const vector<double>& inputs);
	void BackwardPassOutputLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
								 vector<double> expectedOutputs, CostFunction &cost);
	void BackwardPassHiddenLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
								 Layer &nextLayer);
	void ApplyGradientsToWeights(double scalingFactor);
private:
	int numberOfInputs;
	int numberOfNeurons;

	ActivationFunction& activationFunction;
	vector<Neuron> neurons;
};
