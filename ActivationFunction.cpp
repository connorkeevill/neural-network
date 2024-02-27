#include <cmath>
#include <iostream>
#include <utility>
#include "ActivationFunction.h"

vector<double> minMaxScale(vector<double> values)
{
	double min = *min_element(values.begin(), values.end());
	double max = *max_element(values.begin(), values.end());

	for(double &value : values)
	{
		value = (value - min) / (max - min);
	}

	return values;
}


/**
 * @brief Sigmoid function.
 *
 * This function calculates the sigmoid value for the given input.
 *
 * @param neuronOutputs The input value.
 * @return The sigmoid value.
 */
double Sigmoid::Function(vector<double> neuronOutputs, int neuronIndex)
{
	return 1.0 / (1.0 + std::exp(-neuronOutputs[neuronIndex]));
}


/**
 * Calculates the derivative of the sigmoid function at a given input.
 *
 * @param neuronOutputs The input value to the sigmoid function.
 * @return The derivative value of the sigmoid function at the given input.
 */
double Sigmoid::Derivative(vector<double> neuronOutputs, int neuronIndex)
{
	double sigmoid = Function(neuronOutputs, neuronIndex);

	double output = sigmoid * (1 - sigmoid);

	return output;
}

double ReLU::Function(vector<double> neuronOutputs, int neuronIndex)
{
	return std::max(0.0, neuronOutputs[neuronIndex]);
}

double ReLU::Derivative(vector<double> neuronOutputs, int neuronIndex)
{
	return neuronOutputs[neuronIndex] > 0 ? 1 : 0;
}

double Softmax::Function(vector<double> neuronOutputs, int neuronIndex)
{
	double expSum = 0;

	for (int i = 0; i < neuronOutputs.size(); i++)
	{
		expSum += exp(neuronOutputs[i]);
	}

	return exp(neuronOutputs[neuronIndex]) / expSum;
}


double Softmax::Derivative(vector<double> neuronOutputs, int neuronIndex)
{
	double sum = 0;

	for (int i = 0; i < neuronOutputs.size(); i++) {
		sum += exp(neuronOutputs[i]);
	}

	double ex = exp(neuronOutputs[neuronIndex]);
	return 10 * ex * (sum - ex) / (sum * sum);
}
