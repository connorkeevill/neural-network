#pragma once

#include <vector>

using namespace std;

class ActivationFunction
{
public:
	virtual double Function(vector<double> neuronOutputs, int neuronIndex) { return double{}; };
	virtual double Derivative(vector<double> neuronOutputs, int neuronIndex) { return double{}; };
    virtual ~ActivationFunction() = default;
};

class Sigmoid : public ActivationFunction
{
public:
	double Function(vector<double> neuronOutputs, int neuronIndex) override;
	double Derivative(vector<double> neuronOutputs, int neuronIndex) override;
};

class ReLU : public ActivationFunction
{
public:
	double Function(vector<double> neuronOutputs, int neuronIndex) override;
	double Derivative(vector<double> neuronOutputs, int neuronIndex) override;
};

class Softmax : public ActivationFunction
{
public:
  double Function(vector<double> neuronOutputs, int neuronIndex) override;
  double Derivative(vector<double> neuronOutputs, int neuronIndex) override;
};