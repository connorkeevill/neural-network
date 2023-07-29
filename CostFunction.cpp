#include "CostFunction.h"


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
