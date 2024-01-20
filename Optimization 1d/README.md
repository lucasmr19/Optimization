# 1D Optimization Algorithms in Python

This code file contains basic Python implementations of optimization algorithms for finding roots and minima of one-dimensional functions.
**Note:** It uses the NumPy library so make sure you have it  installed before running the code:
```bash
pip install numpy
```
The implemented algorithms include:

1. **Bisection Method:** Finds a root of the function $f$ within the interval $[a, b]$ using the bisection method.

2. **False Position Method:** Finds a root of the function $f$ within the interval $[a, b]$ using the false position method.

3. **Secant Method:** Finds a root of the function $f$ within the interval $[a, b]$ using the secant method.

4. **Trisection Method:** Finds a minimum of the function $f$ within the given bracket using the trisection method.

5. **Golden Section Method:** Finds a minimum of the function $f$ within the given bracket using the golden section method.

6. **Utility Functions:**
   - `root_bracket`: Finds an interval where the function \(f\) changes sign.
   - `min_bracket`: Finds a bracket for the minimum of the function \(f\).

### Usage
```python
import numpy as np
import matplotlib.pyplot as plt
from opt_1d import bisection

# Define a function
def x_sin_exp(x):
 return 0.5 + -(x + np.sin(x)) * np.exp(-x**2.0)

# Example of using the bisection method in the function
a,b, = -3,3 #Choose the interval

# 1: Graph the function to make sure in [a,b] there's only one root
x_data = np.linspace(-3,3, 1000)
y_data = 0.5 -(x_data + np.sin(x_data)) * np.exp(-x_data**2.0)
plt.figure(figsize=(10,6))
plt.title('$f(x) = 0.5 -(x + sin(x))e^{-x^2}$')
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.xticks(np.linspace(-3,3,20),rotation = 45)
plt.plot(x_data, y_data, label ='$f(x)$')
plt.plot(0.2706456263003202, 0, marker = 'x', label = '$x_1$' )
plt.plot(1.2057799467154835 , 0, marker = 'x', label = '$x_2$' )
plt.grid()
plt.axhline(color='black')
plt.legend()
plt.show()

#2. Use the method:
a,b = 0.16,0.5 # Interval with only one root
root, iterations = bisection(f, a, b)
print(f"Root found: {root} after {iterations} iterations.")
```

### Exceptions
- `IterationError`: Raised if the maximum number of iterations is exceeded.
- `ValueError`: Raised if no root is found in the given interval.
