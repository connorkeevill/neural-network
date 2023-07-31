#include "MultilayerPerceptron.h"

#include <utility>
#include <iostream>

/**
 * Construct the network.
 *
 * @param shape a vector of integers specifying both the number of layers, and the number of neurons in each.
 * @param activationFunction the activation function to use. TODO: For now this is just going to be sigmoid.
 */
MultilayerPerceptron::MultilayerPerceptron(const vector<int>& shape, ActivationFunction& activationFunction,
										   CostFunction& costFunction) : costFunction(costFunction)
{
	this->shape = shape;

	for(int layer = 1; layer < shape.size(); ++layer)
	{
		// emplace_back (as opposed to push_back) is slightly more memory efficient as it's in-place.
		this->layers.emplace_back(shape[layer - 1], shape[layer], activationFunction);
	}
}

/**
 * Perform a forward pass of the network
 *
 * @return
 */
vector<double> MultilayerPerceptron::ForwardPass(vector<double> input) {
	vector<double> output = std::move(input);

	for (Layer& layer : this->layers) {
		output = layer.ForwardPass(output);
	}

	return output;
}

/**
 * Gets the number of inputs to the network by examining the first element of the shape vector.
 *
 * @return the number of inputs that the network takes.
 */
int MultilayerPerceptron::NumberOfInputs() {
	return this->shape[0];
}

/**
 * Gets the number of outputs to the network by examining the final element of the shape vector.
 *
 * @return the number of outputs that the network will produce.
 */
int MultilayerPerceptron::NumberOfOutputs() {
	return this->shape[this->shape.size()];
}

void MultilayerPerceptron::Train(Dataset* dataset, double learningRate, int batchSize, int epochs)
{
	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		std::cout << "Epoch: " << epoch << std::endl;

		int trainingExamplesSeen = 0;
		double cost = 0;

		dataset->ResetCounter();
		while(!dataset->EndOfData())
		{
			if(trainingExamplesSeen == batchSize)
			{
				for(Layer& layer : layers)
				{
					layer.ApplyGradientsToWeights(learningRate / trainingExamplesSeen);
				}

				trainingExamplesSeen = 0;
			}

			FeatureVector fv = dataset->GetNextFeatureVector();
			++trainingExamplesSeen;

			vector<double> predicted = ForwardPass(fv.data);
			cost += costFunction.Cost(predicted, fv.label);

			// Handle the network only having two layers (i.e. no hidden layers)
			vector<double> previousActivations = ((int)layers.size()) - 2 >= 0 ? layers[layers.size() - 2].Activations() : fv.data;
			layers[layers.size() - 1].BackwardPassOutputLayer(previousActivations,
															  fv.label,
															  costFunction);

			for(int layer = (int)layers.size() - 2; layer >= 0; --layer)
			{
				previousActivations = layer > 0 ? layers[layer - 1].Activations() : fv.data;
				layers[layer].BackwardPassHiddenLayer(previousActivations, layers[layer + 1]);
			}
		}

		cout << "Batch-averaged cost: " << cost / 60000 << endl;
	}
}
