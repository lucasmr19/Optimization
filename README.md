# 1D Algorithms in Python

This code file contains Python implementations of algorithms for finding roots and minima of one-dimensional functions. The implemented algorithms include:

1. **Bisection Method:**
   - Function `bisection(f, a, b, tol=0.001, maxiter=100)`.
   - Finds a root of the function \(f\) within the interval \([a, b]\) using the bisection method.

2. **False Position Method:**
   - Function `regula_falsi(f, a, b, tol=0.001, maxiter=100)`.
   - Finds a root of the function \(f\) within the interval \([a, b]\) using the false position method.

3. **Secant Method:**
   - Function `secant(f, a, b, tol=0.001, maxiter=100)`.
   - Finds a root of the function \(f\) within the interval \([a, b]\) using the secant method.

4. **Trisection Method:**
   - Function `trisection(f, bracket, xtol=0.001, maxiter=100)`.
   - Finds a minimum of the function \(f\) within the given bracket using the trisection method.

5. **Golden Section Method:**
   - Function `golden(f, bracket, xtol=0.001, maxiter=100)`.
   - Finds a minimum of the function \(f\) within the given bracket using the golden section method.

6. **Utility Functions:**
   - `root_bracket(f, a, b, delta=1.)`: Finds an interval where the function \(f\) changes sign.
   - `min_bracket(f, a, b, delta=1.0)`: Finds a bracket for the minimum of the function \(f\).

### Usage
```python
import numpy as np
from algorithms_1d import *

# Define the function
def f(x):
    return x**2 - 4

# Example of using the bisection method
a, b = root_bracket(f, -3, 3)
root, iterations = bisection(f, a, b)
print(f"Root found: {root} after {iterations} iterations.")
```

### Exceptions
- `IterationError`: Raised if the maximum number of iterations is exceeded.
- `ValueError`: Raised if no root is found in the given interval.

**Note:** Make sure to have the NumPy library installed before running the code:

```bash
pip install numpy
```
