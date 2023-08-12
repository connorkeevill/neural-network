#include "MultilayerPerceptron.h"

/**
 * Construct the network.
 *
 * @param shape a vector of integers specifying both the number of layers, and the number of neurons in each.
 * @param hiddenLayerActivationFunction the activation function to use.
 */
MultilayerPerceptron::MultilayerPerceptron(const vector<int>& shape,
										   ActivationFunction& hiddenLayerActivationFunction,
										   ActivationFunction& outputLayerActivationFunction,
										   CostFunction& costFunction) : costFunction(costFunction)
{
	this->shape = shape;

	for(int layer = 1; layer < shape.size(); ++layer)
	{
		// emplace_back (as opposed to push_back) is slightly more memory efficient as it's in-place.

		if(layer == shape.size() - 1) {
			this->layers.emplace_back(shape[layer - 1], shape[layer], outputLayerActivationFunction);
		}
		else {
			this->layers.emplace_back(shape[layer - 1], shape[layer], hiddenLayerActivationFunction);
		}
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
 * A forward pass, for training the network. Returns the activations for *every* layer, instead of just the last.
 *
 * @param input the input to propagate through the network.
 * @return 2D vector containing the activations of every neuron in the network.
 */
vector<vector<double>> MultilayerPerceptron::TrainingForwardPass(vector<double> input) {
	vector<vector<double>> activations;
	activations.reserve(layers.size());

	vector<double> output = std::move(input);

	for(Layer& layer : layers)
	{
		output = layer.ForwardPass(output);
		activations.push_back(output);
	}

	return activations;
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
	// Create threadpool to train in parallel.
	ThreadPool threads {(size_t)(thread::hardware_concurrency())};
//	ThreadPool threads {1};
	vector<future<void>> tasks {};

	for(int epoch = 0; epoch < epochs; ++epoch)
	{
		int samplesProcessed = 0;
//		cout << "Epoch: " << epoch << endl;

		int batchCount = 0;

		dataset->ResetCounter();
		while(!dataset->EndOfData())
		{
//			cout << "Epoch: " << epoch << " Batch: " << batchCount++ << endl;

			for(int trainingSample = 0; trainingSample < batchSize && samplesProcessed < dataset->Size(); ++trainingSample)
			{
				++samplesProcessed;

				// Pull the feature vector from the dataset now (instead of in thread) so that dataset counter is incremented.
				FeatureVector fv = dataset->GetNextFeatureVector();

				tasks.push_back(threads.enqueue([this, fv] {
					vector<vector<double>> activations = TrainingForwardPass(fv.data);

					// Handle the network only having two layers (i.e. no hidden layers)
					vector<double> previousActivations = ((int)layers.size()) - 2 >= 0 ?
							activations[layers.size() - 2] :
							fv.data;
					vector<double> partialDerivatives = layers[layers.size() - 1].BackwardPassOutputLayer(previousActivations,
																	  activations[layers.size() - 1],
																	  fv.label,
																	  costFunction);

					for(int layer = (int)layers.size() - 2; layer >= 0; --layer)
					{
						previousActivations = layer > 0 ? activations[layer - 1] : fv.data;
						partialDerivatives = layers[layer].BackwardPassHiddenLayer(previousActivations,
															  activations[layer],
															  partialDerivatives,
															  layers[layer + 1]);
					}
				}));
			}

			// Wait on all training tasks to finish
			for(future<void> &task : tasks) { task.get();}
			tasks.clear();

			// Apply the accumulated gradients.
			for(Layer &layer : layers) { tasks.push_back(threads.enqueue([&layer, learningRate, batchSize] {
					layer.ApplyGradientsToWeights(learningRate / batchSize);
				}));
			}

			// Wait on all updates to finish
			for(future<void> &task : tasks) { task.get();}
			tasks.clear();
		}
	}
}
