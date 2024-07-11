# Multidimensional Optimization Module

This Python module implements various optimization algorithms for finding minima of N-dimensional functions. It includes gradient-based methods like gradient descent, Newton's method, Nesterov's accelerated gradient, and quasi-Newton methods (DFP and BFGS). Additionally, each method is extended to include line search variants to dynamically adjust the step size.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Supported Methods](#supported-methods)
5. [Examples](#examples)
6. [References](#references)

---

## Introduction

This module provides a collection of optimization algorithms designed to minimize N-dimensional objective functions. Each algorithm employs different strategies to iteratively converge towards a local minimum.

---

## Installation

To use this module, ensure you have Python 3.6+ installed. You can install the necessary dependencies via pip:

```bash
pip install autograd numpy scipy
```

---

## Usage

Import the module and use the provided functions to minimize your objective function. Each function typically requires:

- `x0`: Initial guess for the minimum point.
- `f`: Objective function to minimize.
- `gf`: Gradient (first-order derivative) of the objective function.

---

## Supported Methods

The module supports the following optimization methods:

- **Gradient Descent**: `grad_desc`
- **Gradient Descent with Line Search**: `grad_desc_ls`
- **Newton's Method**: `newton`
- **Newton's Method with Line Search**: `newton_ls`
- **Nesterov's Accelerated Gradient**: `nesterov_grad_nd`
- **DFP (Davidon-Fletcher-Powell)**: `dfp`
- **DFP with Line Search**: `dfp_ls`
- **FR (Fletcher-Reeves)**: `fr`
- **FR with Line Search**: `fr_ls`

---

## Examples

### Example 1: Gradient Descent

```python
import numpy as np
from optimization_module import grad_desc, f_rosen_md

x0 = np.array([0.5, 0.5])
min_point, points_visited, iterations = grad_desc(x0, f_rosen_md, gf, lr=0.01, maxiter=100)
print(f"Minimum point: {min_point}")
print(f"Iterations: {iterations}")
```

### Example 2: Newton's Method

```python
import numpy as np
from optimization_module import newton, f_rosen_md, gf, hf

x0 = np.array([0.5, 0.5])
min_point, points_visited, iterations = newton(x0, f_rosen_md, gf, hf, lr=0.01, maxiter=100)
print(f"Minimum point: {min_point}")
print(f"Iterations: {iterations}")
```

---

## References

- [Autograd Documentation](https://github.com/HIPS/autograd)
- [SciPy Optimization Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html)

---
