#include "Dataset.h"
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

/**
 * @brief Converts a 32-bit integer from big-endian to host byte order.
 *
 * I am an AI language model created by OpenAI. I have generated this function
 * to take a 32-bit integer `value` and convert it from big-endian byte order
 * (most significant byte first) to the host byte order (the native byte order
 * of the system). It is primarily used for handling data read from files or
 * network protocols where data may be stored in big-endian format.
 *
 * @param value The 32-bit integer value to be converted.
 * @return The value after converting to the host byte order.
 *
 * @note The Function assumes that the input `value` is in big-endian format.
 *       If the input value is already in the host byte order, the Function will
 *       still produce the same value.
 * @note This Function is not available in standard C++ libraries. It's provided
 *       here as a custom implementation and should be used on non-Unix systems
 *       or systems where the `be32toh` Function is not available.
 */
uint32_t be32toh(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8) |
           ((value & 0x0000FF00) << 8) |
           ((value & 0x000000FF) << 24);
}

/**
 * Construct the dataset by setting the counter to be 0. This counter keeps track of how far through the dataset
 * GetNextFeatureVector() has gotten.
 */
Dataset::Dataset()
{
	counter = 0;
}

/**
 * Returns the next datapoint in the dataset, by keeping track of the "counter" of the current position.
 *
 * @return the next datapoint.
 */
FeatureVector Dataset::GetNextFeatureVector()
{
	if(EndOfData()) {throw out_of_range("End of data");}

	FeatureVector current {data[counter], labels[counter]};

	++counter;
	return current;
}

/**
 * Determines whether or not we have reached the end of the dataset.
 *
 * @return boolean.
 */
bool Dataset::EndOfData()
{
	return counter >= data.size();
}

/**
 * Assigns the value 0 to the counter which keeps track of the current datapoint.
 */
void Dataset::ResetCounter(){
	counter = 0;
}

/**
 * @brief Gets the size of the dataset.
 *
 * This function returns the number of items in the dataset.
 *
 * @return The size of the dataset.
 */
int Dataset::Size() {
	return data.size();
}

/**
 * Initialise the MnistDataset by reading in the images and the labels.
 * @param dataFilepath the path to the file containing the images.
 * @param labelsFilepath the path to the file containing the labels.
 */
MnistDataset::MnistDataset(const string& dataFilepath, const string& labelsFilepath) : Dataset()
{
	this->data = ReadDataset(dataFilepath);
	this->labels = ReadLabels(labelsFilepath);
}

/**
 * @brief Converts the output vector into a string representation of the classification.
 *
 * This function takes an output vector, which represents the probabilities of each class, and converts it into a string
 * representation of the classification.
 * The output vector should have the same size as the number of classes in the dataset. The classification string
 * represents the class with the highest probability.
 *
 * @param output The output vector containing the probabilities of each class.
 * @return The string representation of the classification.
 *
 * @note The length of the output vector should be the same as the number of classes in the dataset.
 */
string MnistDataset::ClassificationToString(vector<double> output)
{
	if(output.size() != labels[0].size()) { throw invalid_argument("Input to ClassificationToString() should be same"
																   "length as number of classes in dataset."); }

	double max = -1;
	int maxIndex = 0;

	for(int index = 0; index < output.size(); ++index)
	{
		if(output[index] > max)
		{
			maxIndex = index;
			max = output[index];
		}
	}

	vector<string> classificationStrings {"Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
	return classificationStrings[maxIndex];
}

/**
 * @brief Reads the MNIST dataset from the specified file.
 *
 * This code (and comment) is generated by ChatGPT. This Function reads the MNIST dataset
 * from the provided `filepath`. The dataset is stored in a binary file format and is
 * expected to follow the IDX file format for MNIST data. The Function reads the file,
 * normalizes the pixel values to the range [0, 1], and returns the dataset as a vector
 * of vectors of doubles, where each inner vector represents a 784 (28x28) element vector
 * denoting the pixel values of an image.
 *
 * @param filepath The path to the MNIST dataset file.
 * @return A vector of vectors, where each inner vector is a 784-element vector of doubles
 *         denoting the pixel values of an image.
 *
 * @note The Function assumes that the data file at the given `filepath` is in the correct
 *       MNIST format (IDX file format).
 * @note This Function uses the `be32toh` function for converting big-endian integers to
 *       the host byte order, which might not be available in standard C++ libraries. It's
 *       provided as a custom implementation here for non-Unix systems or systems without
 *       native `be32toh` support.
 */
vector<vector<double>> MnistDataset::ReadDataset(const string& filepath)
{
	vector<vector<double>> mnistData;
    ifstream file(filepath, ios::binary);

    if (!file) {
        cerr << "Error opening file: " << filepath << endl;
        return mnistData;
    }

    // Read the IDX file format information
    uint32_t magicNumber, numImages, numRows, numCols;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    magicNumber = be32toh(magicNumber);
    numImages = be32toh(numImages);
    numRows = be32toh(numRows);
    numCols = be32toh(numCols);

    if (magicNumber != 2051) {
        cerr << "Invalid magic number. This is not a MNIST data file." << endl;
        return mnistData;
    }

    const unsigned int imageSize = numRows * numCols;
    mnistData.reserve(numImages);

	// Read in each image.
    for (uint32_t i = 0; i < numImages; ++i) {
        vector<double> imageData(imageSize);
        vector<uint8_t> buffer(imageSize);

        file.read(reinterpret_cast<char*>(buffer.data()), imageSize);

        for (int j = 0; j < imageSize; ++j) {
            imageData[j] = static_cast<double>(buffer[j]) / 255.0; // Normalize pixel values to [0, 1]
        }

        mnistData.push_back(std::move(imageData));
    }

    return mnistData;
}

/**
 * @brief Reads the MNIST labels from the specified file.
 *
 * This code (and comment) is generated by ChatGPT. This Function reads the MNIST labels
 * from the provided `filepath`. The labels file is stored in a binary format and is
 * expected to follow the IDX file format for MNIST labels. The Function reads the file,
 * converts the labels into one-hot encoded vectors, and returns the labels as a vector
 * of one-hot encoded vectors, where each inner vector represents a 10-element vector
 * denoting the one-hot encoding of a label.
 *
 * @param filepath The path to the MNIST labels file.
 * @return A vector of one-hot encoded vectors, where each inner vector is a 10-element
 *         vector of doubles denoting the one-hot encoding of a label.
 *
 * @note The function assumes that the labels file at the given `filepath` is in the
 *       correct MNIST format (IDX file format).
 * @note This function uses the `be32toh` function for converting big-endian integers to
 *       the host byte order, which might not be available in standard C++ libraries. It's
 *       provided as a custom implementation here for non-Unix systems or systems without
 *       native `be32toh` support.
 */
vector<vector<double>> MnistDataset::ReadLabels(const string& filepath)
{
	vector<vector<double>> mnistLabels;
    ifstream file(filepath, ios::binary);

    if (!file) {
        cerr << "Error opening file: " << filepath << endl;
        return mnistLabels;
    }

    // Read the IDX file format information
    uint32_t magicNumber, numLabels;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numLabels), 4);

    magicNumber = be32toh(magicNumber);
    numLabels = be32toh(numLabels);

    if (magicNumber != 2049) {
        cerr << "Invalid magic number. This is not a MNIST label file." << endl;
        return mnistLabels;
    }

    mnistLabels.reserve(numLabels);

    for (uint32_t i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read(reinterpret_cast<char*>(&label), 1);

        vector<double> oneHotLabel(10, 0.0);
        oneHotLabel[label] = 1.0;

        mnistLabels.push_back(move(oneHotLabel));
    }

    return mnistLabels;
}