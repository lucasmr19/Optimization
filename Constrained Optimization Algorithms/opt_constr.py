

from typing import Callable, List, Tuple
import numpy as np
from scipy.optimize import minimize

def penalty_method(x: np.array, f: Callable, equality_constraints: Callable = None,
                   inequality_constraints: Callable = None, maxiter=100, tol=1e-6,
                   p = 2, lr=1, lr_incr=2, method = 'BFGS') -> Tuple[np.array, List[np.array], int]:
    """
    Implement the Penalty Method for Constrained Optimization. It utilizes the `minimize` function
    from `SciPy` to solve the unconstrained minimization problem, using the optimization
    `method` specified in the minimize function. For more details of this function:
    See here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Parameters:
    -----------
    - `x` (np.array): Initial guess for the optimization.
    - `f` (Callable): Objective function to be minimized.
    - `equality_constraints` (Callable, optional): Function representing equality constraints ``h_i(x) = 0``.
    - `inequality_constraints` (Callable, optional): Function representing inequality constraints ``g(x) <= 0``.
    - `maxiter` (int, optional): Maximum number of iterations.
    - `tol` (float, optional): Tolerance for convergence.
    - `p` (int, optional): Exponent for the penalty term, default is 2.
    - `lr` (float, optional): Initial penalty parameter.
    - `lr_incr` (float, optional): Penalty parameter increment factor.
    - `method` (str, optional): Optimization method used for unconstrained optimization.

    Returns:
    --------
    - `Tuple[np.array, List[np.array], int]`:
        - `x` (np.array): optimal point found.
        - `intermediate_points` (List[np.array]): list of intermediate points.
        - `nit` (int): number of iterations.

    Example:
    --------
    ```python
    def objective_function(x):
        return np.sin(x[0]) + np.cos(x[1]) + x[2]**2 + np.exp(x[3]) - np.log(x[4] + 1)

    def equality_constraints(x):
        return np.array([np.sin(x[0]) + np.cos(x[1]) - x[2]**2, x[0] + x[3] - 2])

    def inequality_constraints(x):
        return np.array([np.exp(x[3]) + x[4] - 5, x[2] + x[4] - 3])

    def test_penalty_method():
        initial_point = np.random.randint(low=0, high=3, size=5)  # random initial point

        result, intermediate_points, iterations = penalty_method(
            initial_point, objective_function, equality_constraints, inequality_constraints
        )

        print("Result:", result)
        print("Iterations:", iterations)
        print("Value in the objective function:", objective_function(result))

    if __name__ == "__main__":
        test_penalty_method()
    ```
    """

    nit = 0
    intermediate_points = [x]

    def _penalized_objective(x):
        penalty_eq = 0 if equality_constraints is None else np.sum(np.abs(equality_constraints(x))**p)
        penalty_ineq = 0 if inequality_constraints is None else np.sum(np.maximum(0, inequality_constraints(x))**p)
        return f(x) + lr * (penalty_eq + penalty_ineq)

    while nit < maxiter:
        # Use unconstrained optimization to minimize the penalty function:
        result = minimize(_penalized_objective, x, method= method)

        x_new = result.x
        intermediate_points.append(x_new)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        lr *= lr_incr
        nit += 1

    return x, intermediate_points, nit
