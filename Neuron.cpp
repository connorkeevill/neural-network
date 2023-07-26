#include "Neuron.h"
#include <numeric>

/**
 * Constructor.
 *
 * @param numberOfInputs the number of inputs to this neuron.
 * @param activationFunction the neuron's activation function.
 */
Neuron::Neuron(int numberOfInputs, const function<double(double)>& activationFunction)
{
	this->numberOfInputs = numberOfInputs;
	this->activationFunction = activationFunction;
}

/**
 * Perform a forward pass on an individual neuron, based on the given input.
 *
 * @param input the inputs to the neuron.
 * @return the neuron's activation after performing the forward pass.
 */
double Neuron::ForwardPass(vector<double> input) {
	this->activation = this->activationFunction(std::inner_product(input.begin(), input.end(), this->weights.begin(), double {}));
	return this->activation;
}

/**
 * Gets the activation of this neuron.
 *
 * @return the activation.
 */
double Neuron::GetActivation()
{
	return this->activation
}
