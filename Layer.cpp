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
		this->neurons.emplace_back(numberOfInputs);
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
	vector<double> outputs;
	vector<double> activations;

	// Allocate the amount of memory we'll need for the output activations for in-loop performance.
	outputs.reserve(this->neurons.size());
	activations.reserve(this->neurons.size());

	for (Neuron& neuron: this->neurons) {
		outputs.push_back(neuron.ForwardPass(inputs));
	}

	for(int output = 0; output < outputs.size(); ++output)
	{
		activations.emplace_back(activationFunction.Function(outputs, output));
	}

	return activations;
}

vector<double> Layer::BackwardPassOutputLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
									vector<double> expectedOutputs, CostFunction &cost) {
	vector<double> partialDerivatives;

	for(int neuron = 0; neuron < neurons.size(); ++neuron)
	{
		double costFunctionDerivative = cost.Derivative(currentLayerActivations[neuron], expectedOutputs[neuron]);
		partialDerivatives.push_back(
				neurons[neuron].UpdateGradients(
						previousLayerActivations,
						activationFunction.Derivative(currentLayerActivations, neuron),
						costFunctionDerivative));
	}

	return partialDerivatives;
}

vector<double> Layer::BackwardPassHiddenLayer(vector<double> &previousLayerActivations, vector<double> &currentLayerActivations,
									vector<double> &nextLayerPartialDerivatives, Layer &nextLayer)
{
	vector<double> partialDerivatives;

	for(int updateNeuron = 0; updateNeuron < neurons.size(); ++updateNeuron)
	{
		double weightedPartialDerivative = 0;

		for(int nextNeuron = 0; nextNeuron < nextLayer.neurons.size(); ++nextNeuron)
		{
			weightedPartialDerivative += nextLayer.neurons[nextNeuron].GetWeight(updateNeuron) *
											nextLayerPartialDerivatives[nextNeuron];
		}

		partialDerivatives.push_back(neurons[updateNeuron].UpdateGradients(previousLayerActivations,
													  currentLayerActivations[updateNeuron],
													  weightedPartialDerivative));
	}

	return partialDerivatives;
}

void Layer::ApplyGradientsToWeights(double scalingFactor) {
	for(Neuron& neuron : neurons)
	{
		neuron.ApplyGradientsToWeights(scalingFactor);
	}
}
