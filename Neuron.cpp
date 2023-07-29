#include "Neuron.h"
#include <numeric>
#include <iostream>
#include "helpers.h"

/**
 * Constructor.
 *
 * @param numberOfInputs the number of inputs to this neuron.
 * @param activationFunction the neuron's activation function.
 */
Neuron::Neuron(int numberOfInputs, ActivationFunction& activationFunction) : activationFunction(activationFunction)
{
	this->numberOfInputs = numberOfInputs;
	this->activationFunction = activationFunction;

	this->weights.reserve(numberOfInputs);
	for(int weight = 0; weight < numberOfInputs; ++weight)
	{
		weights.push_back(rand(-1, 1));
	}

	bias = rand(-1, 1);
}

/**
 * Perform a forward pass on an individual neuron, based on the given input.
 *
 * @param input the inputs to the neuron.
 * @return the neuron's activation after performing the forward pass.
 */
double Neuron::ForwardPass(vector<double> input) {
	if(input.size() != numberOfInputs) {
		throw std::invalid_argument(
				"Incorrect number of inputs (" + to_string(input.size()) + ") provided to neuron (expected " +
				to_string(numberOfInputs) + ").");
	}

	this->activation = this->activationFunction.Function(inner_product(input.begin(), input.end(), this->weights.begin(), double {}) + bias);
	return this->activation;
}

/**
 * Gets the activation of this neuron.
 *
 * @return the activation.
 */
double Neuron::GetActivation()
{
	return this->activation;
}
