"""
Module for implementing algorithms for discrete optimization problems,
including solutions for the fractional knapsack problem and the 0-1 knapsack problem.

#### Author:
  ___                                                       _     __      
 /\_ \                                                    /' \  /'_ `\    
 \//\ \   __  __   ___     __      ____   ___ ___   _ __ /\_, \/\ \L\ \   
   \ \ \ /\ \/\ \ /'___\ /'__`\   /',__\/' __` __`\/\`'__\/_/\ \ \___, \  
    \_\ \\ \ \_\ /\ \__//\ \L\.\_/\__, `/\ \/\ \/\ \ \ \/   \ \ \/__,/\ \ 
    /\____\ \____\ \____\ \__/.\_\/\____\ \_\ \_\ \_\ \_\    \ \_\   \ \_\
    \/____/\/___/ \/____/\/__/\/_/\/___/ \/_/\/_/\/_/\/_/     \/_/    \/_/

#### Implemented Algorithms:
- Fractional Knapsack: using a greedy algorithm.
- Fractional Knapsack: using linear programming (LP) with the `cvxpy` library.
- 0-1 Knapsack: using dynamic programming.
- 0-1 Knapsack: using mixed integer programming (MIP) with the libraries
               `cvxpy`, `pyscipopt`.

For more details on how to use each function, refer to the internal documentation
of each one.

*The libraries `cvxpy` and `pyscipopt` need to be downloaded.*

#### References:
- Explanation of the algorithms: https://en.wikipedia.org/wiki/Knapsack_problem
- `cvxpy` library: https://www.cvxpy.org/
- `pyscipopt` library needed for MIP implementation: https://pypi.org/project/PySCIPOpt/
"""
from typing import Tuple, Dict
from collections import OrderedDict
import numpy as np
import cvxpy as cp


def knapsack_fract_greedy(a_weights: np.ndarray, a_values: np.ndarray,
                          bound: float) -> Tuple[float, Dict]:
    """
    Resuelve el problema de la mochila fraccionaria utilizando un enfoque voraz (greedy).

    Parameters:
    ----------
    - `a_weights` (np.ndarray): Un array NumPy que representa los pesos de los elementos.
    - `a_values` (np.ndarray): Un array NumPy que representa los valores de los elementos.
    - `bound` (float): La capacidad máxima de la mochila.

    Returns:
    --------
    - `Tuple[float, Dict]`:
        - `total_value` (float): El valor total de la mochila óptima.
        - `selected_items` (Dict): Un diccionario que indica el peso a tomar de cada elemento.
           Las claves son los índices de los elementos y los valores son los pesos a tomar.

    Example:
    --------
    >>> weights = np.array([10, 20, 30])
    >>> values = np.array([60, 100, 120])
    >>> capacity = 50
    >>> knapsack_fract_greedy(weights, values, capacity)
    (240.0, {0: 10, 1: 20, 2: 20})
    """
    vpw_sorted = np.argsort(a_values / a_weights)[::-1]

    total_value = 0
    remaining_weight = bound
    selected_items = {}

    # Iterar sobre los value per weight ordenados (de mayor a menor)
    for item in vpw_sorted:
        # Tomar el peso completo si cabe en la mochila
        if a_weights[item] <= remaining_weight:
            total_value += a_values[item]
            selected_items[item] = a_weights[item]  # Tomar el peso completo
            remaining_weight -= a_weights[item]
        else:
            # Tomar fracción del elemento si no cabe completo
            fraction = remaining_weight / a_weights[item]
            total_value += fraction * a_values[item]  # Tomar fracción del peso
            selected_items[item] = remaining_weight
            break  # No se pueden añadir más elementos

    # Ordenar el diccionario de items a devolver
    selected_items = dict(OrderedDict(sorted(selected_items.items())))

    return total_value, selected_items


def knapsack_fract_lin_prog(a_weights: np.ndarray, a_values: np.ndarray,
                            bound: float) -> Tuple[float, Dict]:
    """
    Resuelve el problema de la mochila fraccionaria utilizando programación lineal `(LP)`
    gracias al módulo de `cvxpy`. Por defecto, el solver que usará la función es `'ECOS'`.

    Parameters:
    ----------
    - `a_weights` (np.ndarray): Un array NumPy que representa los pesos de los elementos.
    - `a_values` (np.ndarray): Un array NumPy que representa los valores de los elementos.
    - `bound` (float): La capacidad máxima de la mochila.

    Returns:
    -------
    - `Tuple[float, Dict]`:
      - `optimal_value` (float): El valor total óptimo de la mochila.
      - `selected_weights` (Dict): Un diccionario que indica el peso a tomar de cada elemento.
         Las claves son los índices de los elementos y los valores son los
         pesos a tomar.

    Example:
    -------
    >>> weights = np.array([10, 20, 30])
    >>> values = np.array([60, 100, 120])
    >>> capacity = 50
    >>> knapsack_fract_lin_prog(weights, values, capacity)
    (239.99999991011703, {0: 9.999999987531073, 1: 19.999999977952086, 2: 20.00000002379254})

    References:
    -----------
    https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    """
    n = len(a_weights)
    x = cp.Variable(n)

    objective = cp.Maximize(cp.sum(x @ a_values))
    constraints = [cp.sum(x @ a_weights) <= bound, x >= 0, x <= 1]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    # Obtener el valor óptimo y las variables de decisión
    optimal_value = problem.value
    selected_items = {i: x[i].value * a_weights[i] for i in range(n)}

    return optimal_value, selected_items


def knapsack_01_pd_matrix(a_weights: np.ndarray, a_values: np.ndarray,
                          bound: int) -> np.ndarray:
    """
    Construye la matriz para el problema de la mochila 0-1 utilizando programación
    dinámica `(PD)`. Presupone que cada item se puede seleccionar una sóla vez.

    Parameters:
    ----------
    - `a_weights` (np.ndarray): Un array NumPy que representa los pesos de los elementos.
    - `a_values` (np.ndarray): Un array NumPy que representa los valores de los elementos.
    - `bound` (int): La capacidad máxima de la mochila.

    Returns:
    --------
    - `matrix` (np.ndarray): Una matriz bidimensional que representa la matriz de PD.

    Example:
    -------
    >>> weights = np.array([2, 3, 4, 5])
    >>> values = np.array([3, 4, 5, 6])
    >>> capacity = 5
    >>> knapsack_01_pd_matrix(weights, values, capacity)
    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 3., 3., 3., 3.],
           [0., 0., 3., 4., 4., 7.],
           [0., 0., 3., 4., 5., 7.],
           [0., 0., 3., 4., 5., 7.]])
    """
    n = len(a_weights)
    matrix = np.zeros((n + 1, bound + 1), dtype=float)

    for i in range(1, n + 1):
        for w in range(bound + 1):
            if a_weights[i - 1] <= w:
                matrix[i, w] = max(
                    matrix[i - 1, w], a_values[i - 1] + matrix[i - 1, w - a_weights[i - 1]])
            else:
                matrix[i, w] = matrix[i - 1, w]

    return matrix


def knapsack_01_pd(a_weights: np.ndarray, a_values: np.ndarray,
                   bound: int) -> Tuple[float, Dict]:
    """
    Realiza backtracking sobre la matriz PD obtenida en la función `knapsack_01_pd_matrix` para
    devolver la composición óptima de la mochila, este será el valor máximo.

    Parameters:
    ----------
    - `a_weights` (np.ndarray): Un array NumPy que representa los pesos de los elementos.
    - `a_values` (np.ndarray): Un array NumPy que representa los valores de los elementos.
    - `bound` (int): La capacidad máxima de la mochila.

    Returns:
    -------
    - `Tuple[float, Dict]`:
        - `optimal_value` (float): El valor total de la mochila 0-1 óptima.
        - `selected_items` (Dict): Un diccionario que indica si cada elemento debe incluirse o no.
        Las claves son los índices de los elementos y los valores son 1 si el elemento está
        incluido, 0 si no.

    Example:
    -------
    >>> weights = np.array([2, 3, 4, 5])
    >>> values = np.array([3, 4, 5, 6])
    >>> capacity = 5
    >>> knapsack_01_pd(weights, values, capacity)
    (7.0, {0: 1, 1: 1, 2: 0, 3: 0})
    """
    n = len(a_weights)
    matrix = knapsack_01_pd_matrix(a_weights, a_values, bound)

    # Realizar backtracking para determinar qué elementos incluir en la mochila
    selected_items = {}
    w = bound
    for i in range(n, 0, -1):
        if matrix[i, w] != matrix[i - 1, w]:
            selected_items[i - 1] = 1
            w -= a_weights[i - 1]
        else:
            selected_items[i - 1] = 0

    # Ordenar el diccionario utilizando OrderedDict
    selected_items = dict(OrderedDict(sorted(selected_items.items())))
    return matrix[n, bound], selected_items


def knapsack_01_int_prog(a_weights: np.ndarray, a_values: np.ndarray,
                         bound: int) -> Tuple[float, Dict]:
    """
    Resuelve el problema de la mochila 0-1 utilizando programación mixta entera `(MIP)`
    mediante el módulo de `cvxpy`. El `solver` utilizado por la función será `'SCIP'`.
    Se debe descargar la librería de `pyscipopt` ya que este `solver` no viene por
    defecto con la librería `cvxpy`.

    Parameters:
    ----------
    - `a_weights` (np.ndarray): Un array NumPy que representa los pesos de los elementos.
    - `a_values` (np.ndarray): Un array NumPy que representa los valores de los elementos.
    - `bound` (int): La capacidad máxima de la mochila.

    Returns:
    -------
    - `Tuple[float, Dict]`:
      - `optimal_value` (float): El valor total de la mochila 0-1 óptimo.
      - `selected_items` (Dict): Un diccionario que indica si cada elemento debe incluirse o no.
        Las claves son los índices de los elementos y los valores son 1 si el elemento está
        incluido, 0 si no.

    Example:
    -------
    >>> weights = np.array([2, 3, 4, 5])
    >>> values = np.array([3, 4, 5, 6])
    >>> capacity = 5
    >>> knapsack_01_int_prog(weights, values, capacity)
    (7.0, {0: 1, 1: 1, 2: 0, 3: 0})

    References:
    -----------
    - https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver
    - https://www.scipopt.org/
    - https://pypi.org/project/PySCIPOpt/
    """
    n = len(a_weights)
    x = cp.Variable(n, boolean=True)

    objective = cp.Maximize(cp.sum(x @ a_values))
    constraints = [cp.sum(x @ a_weights) <= bound]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCIPY')

    # Obtener el valor óptimo y las variables de decisión
    optimal_value = problem.value
    selected_items = {i: int(x[i].value) for i in range(n)}

    return optimal_value, selected_items
