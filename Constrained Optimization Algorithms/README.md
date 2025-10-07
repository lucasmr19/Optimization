# Constrained Optimization Algorithms

This module contains Python implementations of classical **constrained optimization algorithms**, including the **Penalty Method**, **Barrier Method**, **Augmented Lagrangian Method**, and **Projected Gradient Method**.
These algorithms transform or adapt constrained problems into unconstrained or feasible forms that can be solved using standard optimization techniques.

---

## Overview

Constrained optimization problems involve minimizing an objective function subject to equality and/or inequality constraints.
This module provides simple but flexible implementations of several foundational methods to handle such problems.

---

## Dependencies

* Python 3.8 or higher
* NumPy
* SciPy

Install dependencies using:

```bash
pip install numpy scipy
```

---

## Implemented Methods

### 1. `penalty_method`

Transforms a constrained problem into an unconstrained one by adding a **penalty term** that grows with the amount of constraint violation.
The penalty parameter is gradually increased to enforce feasibility.

### 2. `barrier_method`

Also known as the **Interior Point Method**, it enforces inequality constraints using a **logarithmic barrier** that keeps the optimizer inside the feasible region.
The barrier parameter decreases over time.

### 3. `augmented_lagrangian_method`

Combines penalty terms with **Lagrange multipliers**, leading to more stable convergence and better accuracy near the solution.
Supports both equality and inequality constraints.

### 4. `projected_gradient_method`

A lightweight **gradient-based approach** that projects each gradient step back into the feasible set.
Useful for problems with simple constraints like bound or convex regions.

---

## Comparative Summary

| **Method**               | **Handles Equalities** | **Handles Inequalities** | **Requires Feasible Start** | **Approach Type**              | **Strengths**                      | **Weaknesses**                            |
| ------------------------ | ---------------------- | ------------------------ | --------------------------- | ------------------------------ | ---------------------------------- | ----------------------------------------- |
| **Penalty Method**       | Yes                    | Yes                      | No                          | External penalty reformulation | Simple and flexible                | Sensitive to scaling of penalty parameter |
| **Barrier Method**       | No                     | Yes                      | Yes                         | Interior-point (log barrier)   | Keeps feasibility automatically    | Requires feasible starting point          |
| **Augmented Lagrangian** | Yes                    | Yes                      | No                          | Hybrid penalty + multiplier    | Fast and stable convergence        | More complex to implement                 |
| **Projected Gradient**   | Limited (simple sets)  | Limited (simple sets)    | No                          | Gradient projection            | Efficient for convex feasible sets | Requires explicit projection operator     |

---

## Example Usage (Penalty Method)

```python
import numpy as np
from constrained_optimization import penalty_method

def objective_function(x):
    return np.sin(x[0]) + np.cos(x[1]) + x[2]**2 + np.exp(x[3]) - np.log(x[4] + 1)

def equality_constraints(x):
    return np.array([
        np.sin(x[0]) + np.cos(x[1]) - x[2]**2,
        x[0] + x[3] - 2
    ])

def inequality_constraints(x):
    return np.array([
        np.exp(x[3]) + x[4] - 5,
        x[2] + x[4] - 3
    ])

def test_penalty_method():
    initial_point = np.random.randint(low=0, high=3, size=5)
    result, intermediate_points, iterations = penalty_method(
        initial_point, objective_function, equality_constraints, inequality_constraints
    )
    print("Result:", result)
    print("Iterations:", iterations)
    print("Objective value:", objective_function(result))

if __name__ == "__main__":
    test_penalty_method()
```

---

## Detailed Descriptions

### Penalty Method

The penalized objective function is defined as:

```
h(x) = f(x) + lr * (penalty_eq + penalty_ineq)
```

Where:

* `penalty_eq = sum(|h_i(x)|^p)` for equality constraints
* `penalty_ineq = sum(max(0, g_i(x))^p)` for inequality constraints

The penalty parameter (`lr`) increases at each iteration to push the solution closer to the feasible region.

---

### Barrier Method

The barrier formulation modifies the objective function as:

```
phi(x, mu) = f(x) - mu * sum(log(-g_i(x)))
```

* `mu` is the barrier parameter (decreased each iteration).
* All inequality constraints must initially satisfy `g_i(x) < 0` (feasible start).

This approach maintains feasibility by preventing the optimizer from crossing constraint boundaries.

---

### Augmented Lagrangian Method

The augmented Lagrangian combines penalty and multiplier terms:

```
L(x, λ, ρ) = f(x)
           + sum(λ_i * h_i(x)) + (ρ/2) * sum(h_i(x)^2)
           + sum( max(0, λ_j + ρ * g_j(x))^2 ) / (2ρ)
```

Where:

* `λ_i` and `λ_j` are Lagrange multipliers for equality and inequality constraints, respectively.
* `ρ` is the penalty parameter, increased iteratively.

This method adapts both parameters for faster convergence.

---

### Projected Gradient Method

Each iteration performs a gradient descent step followed by projection onto the feasible set:

```
x_{k+1} = P_C( x_k - α * grad_f(x_k) )
```

Where:

* `P_C` is the projection operator onto the feasible set `C`.
* `α` is the learning rate (step size).

This method is efficient for simple sets where projection is easy (e.g., box constraints).

---

## Extensibility

This module is designed for easy extension.
Future additions could include:

* Sequential Quadratic Programming (SQP)
* Active-Set Methods
* Primal-Dual Interior Point algorithms

---

## Contributing

Contributions are welcome!
Feel free to submit pull requests or open issues on GitHub for feature requests or bug reports.

---

## License

Licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.
