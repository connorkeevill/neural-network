#include <cmath>
#include <iostream>
#include <utility>
#include "ActivationFunction.h"


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
	return sigmoid * (1 - sigmoid);
}


