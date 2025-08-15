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

  Perceptron(size_t inputs, double bias = 1.0)
}