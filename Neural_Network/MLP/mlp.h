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