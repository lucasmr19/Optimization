# Constrained Optimization Algorithms

This module contains basic Python implementations of constrained optimization algorithms.

## Overview

This module provides an implementation of the Penalty Method for constrained optimization. The Penalty Method is an approach used to solve optimization problems with constraints by transforming them into a series of unconstrained problems.

## Dependencies

- Python 3.6 or higher
- NumPy
- SciPy

You can install the required packages using pip:

```bash
pip install numpy
pip install scipy
```

## Functions: 

- `penalty_method`: This function implements the Penalty Method for constrained optimization.
- 

### Example Usage

```python
import numpy as np
from penalty_method import penalty_method

def objective_function(x):
    return np.sin(x[0]) + np.cos(x[1]) + x[2]**2 + np.exp(x[3]) - np.log(x[4] + 1)

def equality_constraints(x):
    return np.array([np.sin(x[0]) + np.cos(x[1]) - x[2]**2, x[0] + x[3] - 2])

def inequality_constraints(x):
    return np.array([np.exp(x[3]) + x[4] - 5, x[2] + x[4] - 3])

def test_penalty_method():
    initial_point = np.random.randint(low=0, high=3, size=5)  # Random initial point

    result, intermediate_points, iterations = penalty_method(
        initial_point, objective_function, equality_constraints, inequality_constraints
    )

    print("Result:", result)
    print("Iterations:", iterations)
    print("Value in the objective function:", objective_function(result))

if __name__ == "__main__":
    test_penalty_method()
```

### Detailed Description

The Penalty Method works by converting a constrained optimization problem into an unconstrained one. The constraints are incorporated into the objective function through a penalty term. The penalty term is scaled by a penalty parameter, which increases iteratively. The algorithm minimizes the penalized objective function using a specified unconstrained optimization method.

The penalized objective function is defined as:

\[ \text{penalized\_objective}(x) = f(x) + \text{lr} \times (\text{penalty\_eq} + \text{penalty\_ineq}) \]

where

\[ \text{penalty\_eq} = \sum |\text{equality\_constraints}(x)|^p \]

\[ \text{penalty\_ineq} = \sum \max(0, \text{inequality\_constraints}(x))^p \]

The algorithm continues to minimize the penalized objective function until convergence criteria are met, which is determined by the tolerance and maximum number of iterations.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
