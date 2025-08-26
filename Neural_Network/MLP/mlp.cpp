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

    // Initialize the error term for each neuron
    d.push_back(vector<double>(layers[i], 0.0));

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
  for (size_t i = 0; i < w_init.size(); i++)
  {
    for (size_t j = 0; j < w_init[i].size(); j++)
    {
      network[i + 1][j].set_weights(w_init[i][j]);
    }
  }

  // Example of w_init
  vector<vector<vector<double>>> w = {
      {{0, 0}, {0, 0}},
      {{0, 0}}};
}

// Run the Network
vector<double> MultiLayerPerceptron::run(vector<double> x)
{
  // Run an Input Forward Through the Neural Network
  // x is a vector with the input values

  // Set the values for the first layer to the given value x, before it was initialized with {0,0}
  values[0] = x;

  for (size_t i = 1; i < network.size(); i++)
  {
    for (size_t j = 0; j < layers[i]; j++)
    {
      values[i][j] = network[i][j].run(values[i - 1]);
    }
  }
  return values.back(); // Return the output of the last layer
}

// Run a single (x,y) pair with the Backpropagation algorithm
double MultiLayerPerceptron::bp(vector<double> x, vector<double> y)
{
  // Step 1: Feed a sample to the Network
  vector<double> outputs = run(x); // Get the Output of Last Neuron for Each Input i.e. ŷ

  // Step 2: Calculate MSE
  double MSE = 0.0;
  vector<double> error;

  // Calculate the error for each output
  for (size_t i = 0; i < y.size(); i++)
  {
    error.push_back(y[i] - outputs[i]); // ŷ - y
    MSE += error[i] * error[i];         // (ŷ - y)^2 i.e. MSE for Each Sample
  }

  MSE /= layers.back(); // Average of MSE or Loss Function i.e. ((ŷ - y)^2) / N, here N is the number of Perceptron in the output layer

  // Step 3: Calculate the Error for the Neurons in the Last Layer
  for (size_t i = 0; i < outputs.size(); i++)
  {
    d.back()[i] = outputs[i] * (1 - outputs[i]) * (error[i]); // Delta for the Output Layer
  }

  // Step 4: Calculate the error term of each unit on each layer from the forward layer
  for (size_t i = network.size() - 2; i > 0; i--) // Error of each Perceptron only of the Hidden Layer, so substract 2 from the total size of network
  {
    for (size_t h = 0; h < network[i].size(); h++)
    {
      double fwd_error = 0.0;
      for (size_t k = 0; k < layers[i + 1]; k++)
      {
        fwd_error += network[i + 1][k].weights[h] * d[i + 1][k];
      }
      d[i][h] = values[i][h] * (1 - values[i][h]) * fwd_error;
    }
  }

  // Step 5 and 6: Calculate the deltas and update the weights
  for (size_t i = 1; i < network.size(); i++) // Goes through the layers
  {
    for (size_t j = 0; j < layers[i]; j++) // Goes through the neurons
    {
      for (size_t k = 0; k < layers[i - 1] + 1; k++) // Goes through the inputs
      {
        double delta;
        if (k == layers[i - 1])
        {
          delta = eta * d[i][j] * bias;
        }
        else
        {
          delta = eta * d[i][j] * values[i - 1][k];
        }
        network[i][j].weights[k] += delta;
      }
    }
  }

  // Reutrn MSE
  return MSE;
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