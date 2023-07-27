#include <random>

double rand(double lower, double upper) {
    // Create a random number generator engine
    std::random_device rd;  // Obtain a random seed from the operating system
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    // Create a distribution with the desired range
    std::uniform_real_distribution<double> distribution(lower, upper);

    // Generate a random number using the distribution and generator
    return distribution(gen);
}