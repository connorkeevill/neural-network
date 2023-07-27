#include "Layer.h"

/**
 * Construct the layer based on the given number of inputs and outputs, assigning each neuron the given activation
 * function.
 *
 * @param numberOfInputs the number of inputs to this layer.
 * @param numberOfOutputs the number of outputs (or neurons) that this layer has.
 * @param activationFunction the non-linear activation function used by the neurons in this layer.
 */
Layer::Layer(int numberOfInputs, int numberOfOutputs, const function<double(double)>& activationFunction)
{
	this->numberOfInputs = numberOfInputs;
	this->numberOfNeurons = numberOfOutputs;

	for(int neuron = 0; neuron < numberOfOutputs; ++neuron)
	{
		this->neurons.emplace_back(numberOfInputs, activationFunction);
	}
}

/**
 * Performs a forward pass on each neuron in this layer and returns a vector of weights (to pass to the next layer)
 *
 * @param inputs the inputs to this layer.
 * @return the weights of the neurons after the forward pass.
 */
vector<double> Layer::ForwardPass(vector<double> inputs)
{
	vector<double> weights;

	// Allocate the amount of memory we'll need for the output weights for in-loop performance.
	weights.reserve(this->neurons.size());

	for (Neuron neuron: this->neurons) {
		weights.push_back(neuron.ForwardPass(inputs));
	}

	return weights;
}