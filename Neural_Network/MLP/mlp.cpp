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

  // Generate Random Numbers and Fill in the Vectors

  generate(weights.begin(), weights.end(), )
}