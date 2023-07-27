#ifndef NEURAL_NETWORK_MULTILAYERPERCEPTRON_H
#define NEURAL_NETWORK_MULTILAYERPERCEPTRON_H

#include <cstdlib>
#include <vector>
#include "Layer.h"

using namespace std;

class Dataset;

class MultilayerPerceptron
{
public:
	MultilayerPerceptron(const vector<int> shape);

	void Train(const Dataset dataset);

	vector<double> ForwardPass(vector<double>);
private:
	const vector<int> shape;

	vector<Layer> layers;
};


#endif //NEURAL_NETWORK_MULTILAYERPERCEPTRON_H
