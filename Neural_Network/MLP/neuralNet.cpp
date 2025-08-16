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
}