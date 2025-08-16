#pragma once

#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>

class Perceptron
{
public:
  std::vector<double> weights;
  double bias;

  // Constructor
  Perceptron(size_t inputs, double bias = 1.0);

  // Run the Perceptron
  double run(std::vector<double> x);

  // Set the Customize Weights if Needed
  void set_weights(std::vector<double> w_init);

  // Sigmoid Activation Function
  double sigmoid(double x);
};

class MultiLayerPerceptron
{
public:
  // Constructor for initilizing layers
  MultiLayerPerceptron(std::vector<size_t> layers, double bias = 1.0, double eta = 0.5);

  // Set custom weights, w_init for weights of 3 perceptron
  void set_weights(std::vector<std::vector<std::vector<double>>> w_init);

  // Display the weights
  void print_weights();

  // Run the MLP
  std::vector<double> run(std::vector<double> x);

  double bp(std::vector<double> x, std::vector<double> y);

  // Attributes

  std::vector<size_t> layers; // Unsigned Integers, Number of Neurons Per Layer, 0 for Input, 2 for Hidden, 1 for Output

  double bias; // Bias
  double eta;  // Learning Rate

  std::vector<std::vector<Perceptron>> network; // Neural Network
  std::vector<std::vector<double>> values;      // Hodl the Output Valuse of the Network
  std::vector<std::vector<double>> d;           // Error Terms for the Neurons
};