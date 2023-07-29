#pragma once

class ActivationFunction
{
public:
	virtual double Function(double input) = 0;
	virtual double Derivative(double input) = 0;
    virtual ~ActivationFunction() = default;
};

class Sigmoid : public ActivationFunction
{
public:
	double Function(double input) override;
	double Derivative(double input) override;
};
