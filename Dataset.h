#pragma once

#include <string>
#include <vector>

using namespace std;

struct FeatureVector
{
	vector<double> data;
	vector<double> label;
};

class Dataset
{
public:
	Dataset();

	FeatureVector GetNextFeatureVector();
	bool EndOfData();
	void ResetCounter();
	int Size();
	virtual string ClassificationToString(vector<double> output) { return ""; }
protected:
	vector<vector<double>> data;
	vector<vector<double>> labels; // One-hot encoded vector of the label. Might make sense for this to be int or even
								   // bool, but double is in fact needed for backprop (specifically to create the cost
								   // function).
	int counter;
};

// See: http://yann.lecun.com/exdb/mnist/
class MnistDataset : public Dataset
{
public:
	explicit MnistDataset(const string& dataFilepath, const string& labelsFilepath);
	string ClassificationToString(vector<double> output);
private:
	static vector<vector<double>> ReadDataset(const string& filepath);
	static vector<vector<double>> ReadLabels(const string& filepath);
};
