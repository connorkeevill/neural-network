#include <cmath>
#include <iostream>
#include "ActivationFunction.h"


/**
 * @brief Sigmoid function.
 *
 * This function calculates the sigmoid value for the given input.
 *
 * @param input The input value.
 * @return The sigmoid value.
 */
double Sigmoid::Function(double input)
{
	return 1.0 / (1.0 + std::exp(-input));
}


/**
 * Calculates the derivative of the sigmoid function at a given input.
 *
 * @param input The input value to the sigmoid function.
 * @return The derivative value of the sigmoid function at the given input.
 */
double Sigmoid::Derivative(double input)
{
	double sigmoid = Function(input);
	return sigmoid * (1 - sigmoid);
}


