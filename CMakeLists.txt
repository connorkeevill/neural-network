cmake_minimum_required(VERSION 3.25)
project(neural_network)

set(CMAKE_CXX_STANDARD 17)

add_executable(neural_network main.cpp MultilayerPerceptron.h Neuron.cpp MultilayerPerceptron.cpp Layer.cpp)
