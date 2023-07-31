#include "Layer.h"

/**
 * Construct the layer based on the given number of inputs and outputs, assigning each neuron the given activation
 * function.
 *
 * @param numberOfInputs the number of inputs to this layer.
 * @param numberOfOutputs the number of outputs (or neurons) that this layer has.
 * @param activationFunction the non-linear activation function used by the neurons in this layer.
 */
Layer::Layer(int numberOfInputs, int numberOfOutputs, ActivationFunction& activationFunction) : activationFunction(activationFunction)
{
	this->numberOfInputs = numberOfInputs;
	this->numberOfNeurons = numberOfOutputs;

	for(int neuron = 0; neuron < numberOfOutputs; ++neuron)
	{
		this->neurons.emplace_back(numberOfInputs, activationFunction);
	}
}

/**
 * Performs a forward pass on each neuron in this layer and returns a vector of activations (to pass to the next layer)
 *
 * @param inputs the inputs to this layer.
 * @return the weights of the neurons after the forward pass.
 */
vector<double> Layer::ForwardPass(const vector<double>& inputs)
{
	vector<double> activations;

	// Allocate the amount of memory we'll need for the output activations for in-loop performance.
	activations.reserve(this->neurons.size());

	for (Neuron& neuron: this->neurons) {
		activations.push_back(neuron.ForwardPass(inputs));
	}

	return activations;
}

void Layer::BackwardPassOutputLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
									vector<double> expectedOutputs, CostFunction &cost) {
	for(int neuron = 0; neuron < neurons.size(); ++neuron)
	{
		double costFunctionDerivative = cost.Derivative(currentLayerActivations[neuron], expectedOutputs[neuron]);
		neurons[neuron].UpdateGradients(previousLayerActivations, currentLayerActivations[neuron],
										costFunctionDerivative);
	}
}

void Layer::BackwardPassHiddenLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
									Layer& nextLayer)
{
	for(int updateNeuron = 0; updateNeuron < neurons.size(); ++updateNeuron)
	{
		double weightedPartialDerivative = 0;

		for(Neuron nextNeuron : nextLayer.neurons)
		{
			weightedPartialDerivative += nextNeuron.GetWeight(updateNeuron) * nextNeuron.PartialDerivative;
		}

		neurons[updateNeuron].UpdateGradients(previousLayerActivations, currentLayerActivations[updateNeuron],
											  weightedPartialDerivative);
	}
}

void Layer::ApplyGradientsToWeights(double scalingFactor) {
	for(Neuron& neuron : neurons)
	{
		neuron.ApplyGradientsToWeights(scalingFactor);
	}
}
