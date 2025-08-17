#include "mlp.h"
#include <iostream>
using namespace std;

// Random Number Generator Function
double frand()
{
  return (2.0 * (double)rand() / RAND_MAX) - 1.0;
}

/*
Single Layer Perceptron Implementation
*/

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

/*
Multi Layer Perceptron Implementation
*/

// Return a new MLP Object with the Specified number layers, bias and Learning Rate

MultiLayerPerceptron::MultiLayerPerceptron(std::vector<size_t> layers, double bias, double eta) : layers(layers), bias(bias), eta(eta)
{
  // Create Neurons Layer By Layer
  // Outer Loop
  for (size_t i = 0; i < layers.size(); i++)
  {
    // Add Vector of Values Filled with Zeros
    values.push_back(vector<double>(layers[i], 0.0)); // Output of Each Neuron Value set to Zero based on the number of Neurons in Each layer

    // Add Vector of Neurons
    network.push_back(vector<Perceptron>()); // Creates a temporary empty std::vector<Perceptron> object and pushes it into 'network'. Without '()', we would only refer to the type name, not an object, causing a compiler error.

    // Inner Loop
    // network[0] is the input layer, so it has no neurons
    if (i > 0)
    {
      // Iterate on Each Neuron in the Layer
      for (size_t j = 0; j < layers[i]; j++)
      {
        // Add Perceptron in Every Layer, Starting with the Layer 1, cause 0 is Input Layer
        // Each Perceptron Should Accept the Input as Number of Neurons in the Pervious Layer
        network[i].push_back(Perceptron(layers[i - 1], bias));
      }
    }
  }
}

// Set Custom Weights
void MultiLayerPerceptron::set_weights(vector<vector<vector<double>>> w_init)
{
  // Write all the weights into the neural network
  // w_init is a vector of vectors of vectors of doubles
}

// Run the Network
vector<double> MultiLayerPerceptron::run(vector<double> x)
{
  // Run an Input Forward Through the Neural Network
  // x is a vector with the input values
}

// Print the Weights
void MultiLayerPerceptron::print_weights()
{
  cout << endl;

  for (size_t i = 1; i < network.size(); i++)
  {
    for (size_t j = 0; j < layers[i]; j++)
    {
      cout << "Layer" << i + 1 << " Neuron " << j << ": ";
      for (auto &it : network[i][j].weights)
      {
        cout << it << " ";
      }
      cout << endl;
    }
  }
}