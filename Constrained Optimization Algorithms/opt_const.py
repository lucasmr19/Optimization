"""
This module contains basic Python implementations of constrained optimization algorithms.

Author: 
%  ___                                                       _     __      
% /\_ \                                                    /' \  /'_ `\    
% \//\ \   __  __   ___     __      ____   ___ ___   _ __ /\_, \/\ \L\ \   
%   \ \ \ /\ \/\ \ /'___\ /'__`\   /',__\/' __` __`\/\`'__\/_/\ \ \___, \  
%    \_\ \\ \ \_\ /\ \__//\ \L\.\_/\__, `/\ \/\ \/\ \ \ \/   \ \ \/__,/\ \ 
%    /\____\ \____\ \____\ \__/.\_\/\____\ \_\ \_\ \_\ \_\    \ \_\   \ \_\
%    \/____/\/___/ \/____/\/__/\/_/\/___/ \/_/\/_/\/_/\/_/     \/_/    \/_/
"""

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

def barrier_method(x: np.array, f: Callable, inequality_constraints: Callable,
                   mu_init=1.0, mu_reduction=0.5, tol=1e-6, maxiter=100, method='BFGS') -> Tuple[np.array, List[np.array], int]:
    """
    Implements the Interior-Point (Barrier) Method for constrained optimization.
    Only handles inequality constraints of the form g_i(x) <= 0.

    The barrier method replaces inequality constraints with a logarithmic barrier term:
        φ(x, μ) = f(x) - μ * Σ log(-g_i(x))

    Parameters
    ----------
    x : np.array
        Initial guess (must satisfy g_i(x) < 0 for all i).
    f : Callable
        Objective function to minimize.
    inequality_constraints : Callable
        Function that returns an array of inequality constraints g_i(x) <= 0.
    mu_init : float
        Initial barrier parameter μ > 0.
    mu_reduction : float
        Factor by which μ is multiplied each iteration (0 < mu_reduction < 1).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of outer iterations.
    method : str
        Optimization method used in SciPy’s `minimize`.

    Returns
    -------
    Tuple[np.array, List[np.array], int]
        - Optimal x
        - List of intermediate points
        - Number of iterations
    """

    mu = mu_init
    nit = 0
    intermediate_points = [x]

    def _barrier_objective(x, mu):
        g = inequality_constraints(x)
        if np.any(g >= 0):  # infeasible -> large penalty
            return np.inf
        return f(x) - mu * np.sum(np.log(-g))

    while nit < maxiter:
        # Minimize the barrier function
        result = minimize(lambda x_: _barrier_objective(x_, mu), x, method=method)
        x_new = result.x
        intermediate_points.append(x_new)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        mu *= mu_reduction  # decrease barrier parameter
        nit += 1

    return x, intermediate_points, nit

def augmented_lagrangian_method(x: np.array, f: Callable,
                                equality_constraints: Callable = None,
                                inequality_constraints: Callable = None,
                                rho_init=1.0, rho_incr=2.0, tol=1e-6,
                                maxiter=100, method='BFGS') -> Tuple[np.array, List[np.array], int]:
    """
    Implements the Augmented Lagrangian Method for constrained optimization.

    The augmented Lagrangian function is:
        L(x, λ, ρ) = f(x)
                     + Σ λ_i * h_i(x) + (ρ/2) * ||h_i(x)||^2
                     + Σ max(0, λ_j + ρ * g_j(x))^2 / (2ρ)

    Parameters
    ----------
    x : np.array
        Initial guess.
    f : Callable
        Objective function.
    equality_constraints : Callable, optional
        Equality constraints h_i(x) = 0.
    inequality_constraints : Callable, optional
        Inequality constraints g_j(x) <= 0.
    rho_init : float
        Initial penalty parameter ρ.
    rho_incr : float
        Multiplicative increase for ρ after each iteration.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.
    method : str
        Optimization method (for SciPy minimize).

    Returns
    -------
    Tuple[np.array, List[np.array], int]
        - Optimal x
        - List of intermediate points
        - Number of iterations
    """

    rho = rho_init
    nit = 0
    intermediate_points = [x]

    # Initialize multipliers
    lam_eq = np.zeros_like(equality_constraints(x)) if equality_constraints else np.array([])
    lam_ineq = np.zeros_like(inequality_constraints(x)) if inequality_constraints else np.array([])

    while nit < maxiter:
        def L(x):
            val = f(x)
            if equality_constraints is not None:
                h = equality_constraints(x)
                val += np.dot(lam_eq, h) + (rho / 2) * np.sum(h ** 2)
            if inequality_constraints is not None:
                g = inequality_constraints(x)
                val += np.sum(np.maximum(0, lam_ineq + rho * g) ** 2) / (2 * rho)
            return val

        result = minimize(L, x, method=method)
        x_new = result.x
        intermediate_points.append(x_new)

        # Update multipliers
        if equality_constraints is not None:
            lam_eq += rho * equality_constraints(x_new)
        if inequality_constraints is not None:
            lam_ineq = np.maximum(0, lam_ineq + rho * inequality_constraints(x_new))

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        rho *= rho_incr
        nit += 1

    return x, intermediate_points, nit

def projected_gradient_method(x: np.array, f: Callable, grad_f: Callable,
                              projection: Callable, step_size=0.01,
                              tol=1e-6, maxiter=1000) -> Tuple[np.array, List[np.array], int]:
    """
    Implements the Projected Gradient Method for simple constraint sets.

    This method assumes the constraints define a convex feasible set,
    and that we can project any point back into it using the function `projection`.

    Algorithm:
        x_{k+1} = P_C(x_k - α * ∇f(x_k))

    Parameters
    ----------
    x : np.array
        Initial guess.
    f : Callable
        Objective function to minimize.
    grad_f : Callable
        Gradient of the objective function.
    projection : Callable
        Function P_C(x) that projects a point onto the feasible set C.
    step_size : float
        Learning rate (α).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    Tuple[np.array, List[np.array], int]
        - Optimal x
        - List of intermediate points
        - Number of iterations
    """

    nit = 0
    intermediate_points = [x]

    for _ in range(maxiter):
        grad = grad_f(x)
        x_new = projection(x - step_size * grad)
        intermediate_points.append(x_new)

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        nit += 1

    return x, intermediate_points, nit
