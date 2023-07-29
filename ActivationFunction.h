#pragma once

class ActivationFunction
{
public:
	virtual double Function(double input) { return double{}; };
	virtual double Derivative(double input) { return double{}; };
    virtual ~ActivationFunction() = default;
};

class Sigmoid : public ActivationFunction
{
public:
	double Function(double input) override;
	double Derivative(double input) override;
};
