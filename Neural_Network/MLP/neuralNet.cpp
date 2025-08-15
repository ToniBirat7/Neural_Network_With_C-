#include "mlp.h"
#include <iostream>
#include <vector>
using namespace std;

int main()
{
  Perceptron p(2); // Object with 2 inputs on the Stack, No need to delete

  p.set_weights({10, 10, -5}); // +1 Bias

  cout << "Gate: AND" << endl;
  cout << "0 AND 0 = " << p.run({0, 0}) << endl;
  cout << "0 AND 1 = " << p.run({0, 1}) << endl;
  cout << "1 AND 0 = " << p.run({1, 0}) << endl;
  cout << "1 AND 1 = " << p.run({1, 1}) << endl;
}