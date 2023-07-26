//
// Created by Connor Keevill on 26/07/2023.
//

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H

#include "Neuron.h"
#include <vector>
#include <functional>

class Layer
{
public:
	Layer(int numberOfInputs, int numberOfOutputs, function<double(double)> activationFunction);

	vector<double> ForwardPass(vector<double> inputs);
private:
	int numberOfInputs;
	int numberOfNeurons;

	vector<Neuron> neurons;
};


#endif //NEURAL_NETWORK_LAYER_H
