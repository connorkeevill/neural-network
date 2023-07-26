#ifndef NEURAL_NETWORK_MULTILAYERPERCEPTRON_H
#define NEURAL_NETWORK_MULTILAYERPERCEPTRON_H

#include <cstdlib>
#include <vector>
#include "Layer.h"

class Dataset;

class MultilayerPerceptron
{
public:
	MultilayerPerceptron(const std::vector<int> size);

	void Train(const Dataset dataset);
};


#endif //NEURAL_NETWORK_MULTILAYERPERCEPTRON_H
