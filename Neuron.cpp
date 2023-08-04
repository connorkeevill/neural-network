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

	this->gradientMutex = new mutex{};

	this->weights.reserve(numberOfInputs);
	this->weightGradients.reserve(numberOfInputs);
	for(int weight = 0; weight < numberOfInputs; ++weight)
	{
		weights.push_back(rand(-1, 1));
		weightGradients.push_back(0);
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

	return this->activationFunction.Function(inner_product(input.begin(), input.end(), this->weights.begin(), double {}) + bias);;
}

/**
 * Returns the weight at the given index.
 *
 * @param index index of the weight wanted.
 * @return the weight.
 */
double Neuron::GetWeight(int index)
{
	return weights[index];
}

double Neuron::UpdateGradients(vector<double> &previousActivations, double activation, double nextLayerPartialDerivative)
{
	double currentPartialDerivative = activationFunction.Derivative(activation) * nextLayerPartialDerivative;

	gradientMutex->lock();
	for(int weightIndex = 0; weightIndex < weights.size(); ++weightIndex)
	{
		weightGradients[weightIndex] += previousActivations[weightIndex] * currentPartialDerivative;
	}

	biasGradient += currentPartialDerivative;
	gradientMutex->unlock();

	return currentPartialDerivative;
}

void Neuron::ApplyGradientsToWeights(double scalingFactor)
{
	for(int weightIndex = 0; weightIndex < weights.size(); ++weightIndex)
	{
		weights[weightIndex] -= scalingFactor * weightGradients[weightIndex];
		weightGradients[weightIndex] = 0;
	}

	bias -= scalingFactor * biasGradient;
	biasGradient = 0;
}
