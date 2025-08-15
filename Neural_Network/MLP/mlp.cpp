#include "mlp.h"
#include <iostream>
using namespace std;

// Random Number Generator Function
double frand()
{
  return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

// Return a new Perceptron Object with the Specified number of Inputs (+1 for the bias)

Perceptron::Perceptron(size_t inputs, double bias)
{
  this->bias = bias;

  // Initialize the Weights as Random numbers of Double between -1 and 1

  weights.resize(inputs + 1); // Resize the Vector for Weights + Bias

  // Generate Random Numbers and Fill in the Vectors. Pass the frand function to generate the number

  generate(weights.begin(), weights.end(), frand);
}

// Run Function
// Feeds an Input Vector X into the perceptron to return the activation function output.

double Perceptron::run(std::vector<double> x)
{

  // Add the bias at the end
  x.push_back(bias);

  // Weighted Sum
  double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);

  return sigmoid(sum); // Pass into the sigmoid function
}

// Set the weights. w_init is a vector with the Weights

void Perceptron::set_weights(std::vector<double> w_init)
{
  weights = w_init; // Copies the vector
}

// Evaluate the Sigmoid Function for the floating point of input

double Perceptron::sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}