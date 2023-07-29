#pragma once

#include <vector>

using namespace std;

class CostFunction
{
public:
	virtual double Cost(vector<double> predicted, vector<double> expected) { return double{}; };
	virtual double Derivative(double predicted, double expected) { return double{}; };
	~CostFunction() = default;
protected:
	void guardAgainstInvalidVectorLengths(vector<double> a, vector<double> b);
};

class MeanSquaredError : public CostFunction
{
public:
	double Cost(vector<double> predicted, vector<double> expected) override;
	double Derivative(double predicted, double expected) override;
};
