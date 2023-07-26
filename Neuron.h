#pragma once

#include <vector>
#include <functional>

using namespace std;

class Neuron {
public:
	Neuron(int numberOfInputs, const function<double(double)>& activationFunction);

	double ForwardPass(vector<double> input);

	double GetActivation();
private:
	int numberOfInputs;
	function<double(double)> activationFunction;

	vector<double> weights;
	double bias;

	double activation;
};
