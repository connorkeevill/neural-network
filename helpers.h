#include <random>
#include <iostream>
#include <fstream>
#include <vector>

double rand(double lower, double upper) {
    // Create a random number generator engine
    std::random_device rd;  // Obtain a random seed from the operating system
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    // Create a distribution with the desired range
    std::uniform_real_distribution<double> distribution(lower, upper);

    // Generate a random number using the distribution and generator
    return distribution(gen);
}

/**
 * Writes the given feature vector (of greyscale pixels values) to a .ppm file.
 * Note that the pixel values should be on the interval [0, 1], as if neuron activations.
 *
 * @param pixels the pixels in the feature vector
 * @param width the width of the image
 * @param height the height of the image
 * @param filename the filename to write the image to; should have .ppm extension
 */
void WriteFeatureVectorAsImage(const std::vector<double>& pixels, int width, int height, const std::string& filename) {

	// Check if the vector size matches the specified dimensions (28x28 = 784)
    if (pixels.size() != width * height) {
        std::cerr << "Invalid input vector size. Expected 784 elements (28x28 pixels)." << std::endl;
        return;
    }

    // Open the file in binary mode for writing
    std::ofstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write the PPM header
    file << "P5\n" << width << " " << height << "\n255\n";

    // Write the pixel values to the file
	for(double P : pixels)
	{
		P *= 255; // Multiply to scale up to correct value.
		file << (unsigned char) P;
	}

    // Close the file
    file.close();

    std::cout << "Image data saved to: " << filename << std::endl;
}
