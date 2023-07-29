#include "CostFunction.h"


void CostFunction::guardAgainstInvalidVectorLengths(vector<double> a, vector<double> b)
{
	if (a.size() != b.size()) {throw invalid_argument("Invalid call to cost function: vectors have different size."); }
}

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

double MeanSquaredError::Derivative(double predicted, double expected)
{
	return 2 * (predicted - expected);
}
