#include "CostFunction.h"
#include <iostream>


/**
 * @brief Guard against invalid vector lengths.
 *
 * This function checks if the lengths of two input vectors `a` and `b` are equal.
 * If the lengths are not equal, it throws an exception indicating the invalid vector lengths.
 *
 * @param a First input vector of type `std::vector<double>`.
 * @param b Second input vector of type `std::vector<double>`.
 * @throws std::invalid_argument if the lengths of `a` and `b` are not equal.
 */
void CostFunction::guardAgainstInvalidVectorLengths(vector<double> a, vector<double> b)
{
	if (a.size() != b.size()) {throw invalid_argument("Invalid call to cost function: vectors have different size."); }
}

/**
 * @class MeanSquaredError
 * @brief Calculates the mean squared error between predicted and expected values.
 *
 * @param predicted the values estimated by the network.
 * @param expected the actual values as given by the label.
 */
double MeanSquaredError::Cost(vector<double> predicted, vector<double> expected)
{
	guardAgainstInvalidVectorLengths(predicted, expected);

	double cost = 0;

	for(int index = 0; index < predicted.size(); ++index)
	{
		cost += pow((predicted[index] - expected[index]), 2);
	}

	return cost;
}

/**
 * Returns the derivative of the cost function at the given inputs.
 *
 * @param predicted the predicted value to calculate the derivative at.
 * @param expected the expected value to calculate the derivative at.
 * @return the derivative.
 */
double MeanSquaredError::Derivative(double predicted, double expected)
{
	return 2 * (predicted - expected);
}


/**
 * @class CrossEntropy
 * @brief Calculates the cross entropy loss between predicted and expected values.
 *
 * @param predicted the predicted values by the network.
 * @param expected the actual values as given by the label.
 */
double CrossEntropy::Cost(vector<double> predicted, vector<double> expected)
{
	// cost is sum (for all x,y pairs) of: 0.5 * (x-y)^2
	double cost = 0;

	for (int i = 0; i < predicted.size(); i++)
	{
		double v = (expected[i] == 1) ? -log(predicted[i]) : -log(1 - predicted[i]);
		cost += isnan(v) ? 0 : v;
	}
	return cost;
}

/**
 * Returns the derivative of the cost function at the given inputs.
 *
 * @param predicted the predicted value to calculate the derivative at.
 * @param expected the expected value to calculate the derivative at.
 * @return the derivative.
 */
double CrossEntropy::Derivative(double predicted, double expected)
{

	if (predicted == 0 || predicted == 1)
	{
		return 0;
	}

	double output = (-predicted + expected) / (predicted * (predicted - 1));
	return output;
}
