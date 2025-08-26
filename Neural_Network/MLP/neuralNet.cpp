#include "mlp.h"
#include <iostream>
#include <vector>
using namespace std;

int main()
{
  // AND Gate
  Perceptron andGate(2); // Object with 2 inputs on the Stack, No need to delete

  andGate.set_weights({10, 10, -15}); // +1 Bias

  cout << "Gate: AND" << endl;
  cout << "0 AND 0 = " << andGate.run({0, 0}) << endl;
  cout << "0 AND 1 = " << andGate.run({0, 1}) << endl;
  cout << "1 AND 0 = " << andGate.run({1, 0}) << endl;
  cout << "1 AND 1 = " << andGate.run({1, 1}) << endl;

  // OR Gate
  Perceptron orGate(2); // Object with 2 inputs on the Stack, No need to delete

  orGate.set_weights({15, 15, -10}); // +1 Bias

  cout << "Gate: OR" << endl;
  cout << "0 AND 0 = " << orGate.run({0, 0}) << endl;
  cout << "0 AND 1 = " << orGate.run({0, 1}) << endl;
  cout << "1 AND 0 = " << orGate.run({1, 0}) << endl;
  cout << "1 AND 1 = " << orGate.run({1, 1}) << endl;

  // XOR Gate
  // Instaintiate with 2 inputs, 2 Perceptron in hidden and 1 Perceptron in the Output
  MultiLayerPerceptron mlp({2, 2, 1});

  // Set the Weights, NAND Gate, OR Gate and then AND Gate
  mlp.set_weights(
      {{{-10, -10, 15}, {15, 15, -10}},
       {{10, 10, -15}}});

  cout << endl;

  // Print Weights
  cout << "Hardcoded Weights:" << endl;
  mlp.print_weights();

  cout << endl;

  // Run the Network
  cout << "XOR: " << endl;
  cout << "0 0 = " << mlp.run({0, 0})[0] << endl; // For 0 0 Input, Output should be  0.00669585
  cout << "0 0 = " << mlp.run({0, 1})[0] << endl; // For 0 1 Input, Output should be  1
  cout << "0 0 = " << mlp.run({1, 0})[0] << endl; // For 1 0 Input, Output should be  1
  cout << "0 0 = " << mlp.run({1, 1})[0] << endl; // For 1 1 Input, Output should be  0

  // XOR Gate with Back Propagation
  cout << endl;

  cout << "Training Neural Network as an XOR Gate..." << endl;

  MultiLayerPerceptron mlpBp({2, 2, 1}); // MLP Object

  double MSE; // Mean Square Error

  for (int i = 0; i < 3000; i++)
  {
    MSE = 0.0;
    MSE += mlpBp.bp({0, 0}, {0});
    MSE += mlpBp.bp({0, 1}, {1});
    MSE += mlpBp.bp({1, 0}, {1});
    MSE += mlpBp.bp({1, 1}, {0});

    MSE = MSE / 4.0;

    if (i % 100 == 0)
    {
      cout << "MSE = " << MSE << endl;
    }
  }

  cout << "\n\nTrained Weights (Compare to Hardcoded Weights): \n\n";
  mlp.print_weights();

  cout << "XOR:" << endl;
  cout << "0 0 = " << mlpBp.run({0, 0})[0] << endl;
  cout << "0 1 = " << mlpBp.run({0, 1})[0] << endl;
  cout << "1 0 = " << mlpBp.run({1, 0})[0] << endl;
  cout << "1 1 = " << mlpBp.run({1, 1})[0] << endl;
}