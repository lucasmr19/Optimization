from typing import Tuple, List, Callable
import autograd.numpy as np
from scipy.optimize import minimize_scalar, minimize
from autograd import jacobian
import scipy.optimize

def grad_desc(x: np.array, f: Callable, gf: Callable, lr=0.01, lr_decr=0.999,
              maxiter=100, tol=0.001) -> Tuple[np.array, List[np.array], int]:
    """
    Realiza el descenso por gradiente para minimizar una función cuyo argumento es multidimensional.
    La actualización de la tasa de aprendizaje se realiza según el criterio:
    `tk+1 = tk x p`, con `p` el decaimiento.

    Parameters:
    ----------
        - `x` (np.ndarray): Un array que representa el punto inicial donde empieza el algoritmo.
        - `f` (Callable): Función objetivo a minimizar.
        - `gf` (Callable): Gradiente de la función objetivo.
        - `lr` (float, optional): Tasa de aprendizaje inicial. Por defecto es 0.01.
        - `lr_decr` (float, optional): Factor de decaimiento para la tasa de aprendizaje.
        Por defecto es 0.999.
        - `maxiter` (int, optional): Número máximo de iteraciones. Por defecto es 100.
        - `tol` (float, optional): Tolerancia para la norma del gradiente que determina la
        convergencia. Por defecto es 0.001.

    Returns:
    --------
        - `Tuple[np.ndarray, List[np.ndarray], int]`:
            - El punto aproximado al mínimo.
            - Una lista de puntos intermedios (arrays) calculados durante la optimización.
            - El número de iteraciones realizadas.
    """
    points = [x]
    nit = 0
    gradient = gf(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:

        x = x - lr * gradient
        lr *= lr_decr
        points.append(x)
        nit += 1
        gradient = gf(x)

    return x, points, nit


def f_md(f: Callable, x: np.array, gx: np.array) -> Callable:
    """
    Crea una función unidimensional `f_1d(t)` a partir de una función multidimensional `f`,
    su gradiente y un punto.

    Parameters:
    ----------
        - `f` (Callable): Función objetivo multidimensional.
        - `x` (np.array): Punto multidimensional.
        - `gx` (np.array): Gradiente de la función objetivo en el punto `x`.

    Returns:
    --------
        `Callable`: Una función unidimensional `f_1d(t)` que evalua `f(x - t * gx)`.
    """
    def f_1d(t: float) -> float:
        """
        Evalúa la función multidimensional `f` a lo largo de la dirección especificada por
        el punto `x` y su gradiente `gx`, desplazándose en la dirección opuesta a `gx`
        multiplicado por el escalar `t`.

        Parameters:
        ----------
            `t` (float): Escalar que determina la magnitud del desplazamiento en la dirección `-gx`.

        Returns:
        --------
            `float`: Valor de la función multidimensional `f` evaluada en el punto `x - t * gx`.
        """
        return f(x - t * gx)

    return f_1d


def grad_desc_ls(x: np.array, f: Callable, gf: Callable, maxiter=100, tol=0.001,
                 method='brent') -> Tuple[np.ndarray, List[np.array], int]:
    """
    Realiza el descenso por gradiente de una función cuyo argumento es multidimensional,
    usando además sondeo lineal para hallar el valor óptimo del factor de aprendizaje (`lr_opt`)
    para cada iteración.

    Parameters:
    ----------
        - `x` (np.array): Un array que representa el punto inicial donde empieza el algoritmo.
        - `f` (Callable): Función objetivo a minimizar.
        - `gf` (Callable): Gradiente de la función objetivo.
        - `maxiter` (int, optional): Número máximo de iteraciones. Por defecto es 100.
        - `tol` (float, optional): Tolerancia para la norma del gradiente que determina la
        exactitud del punto calculado. Por defecto es 0.001.
        - `method` (str, optional): Método de optimización para el sondeo lineal (por defecto,
        'brent' del método minimize_scalar de scipy.optimize).

    Returns:
    --------
        - `Tuple[np.ndarray, List[np.array], int]`:
            - El punto aproximado al mínimo.
            - Una lista de puntos intermedios (arrays) calculados durante la optimización.
            - El número de iteraciones realizadas.
    """
    points = [x]
    nit = 0
    gradient = gf(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:

        f_1d = f_md(f, x, gradient)

        res = minimize_scalar(f_1d, method=method)
        lr_opt = res.x

        x = x - lr_opt * gradient

        points.append(x)
        nit += 1
        gradient = gf(x)

    return x, points, nit


def newton(x: np.array, f: Callable, gf: Callable, hf: Callable, lr=0.01, lr_decr=0.999,
           maxiter=100, tol=0.001) -> Tuple[np.ndarray, List[np.array], int]:
    """
    Realiza el método de Newton para minimizar una función cuyo argumento es multidimensional.
    La actualización de la tasa de aprendizaje se realiza según el criterio:
    `tk+1 = tk x p`, con `p` el decaimiento.

    Parameters:
    ----------
        - `x` (np.array): Un array que representa el punto inicial donde empieza el algoritmo.
        - `f` (Callable): Función objetivo a minimizar.
        - `gf` (Callable): Gradiente de la función objetivo.
        - `hf` (Callable): Hessiano de la función objetivo.
        - `lr` (float, optional): Tasa de aprendizaje inicial. Por defecto es 0.01.
        - `lr_decr` (float, optional): Factor de decaimiento para la tasa de aprendizaje.
        Por defecto es 0.999.
        - `maxiter` (int, optional): Número máximo de iteraciones. Por defecto es 100.
        - `tol` (float, optional): Tolerancia para la norma del gradiente que determina la
        convergencia. Por defecto es 0.001.

    Returns:
    --------
        - `Tuple[np.ndarray, List[np.array], int]`:
            - El punto aproximado al mínimo.
            - Una lista de puntos intermedios (arrays) calculados durante la optimización.
            - El número de iteraciones realizadas.
    """
    points = [x]
    nit = 0
    gradient = gf(x)
    hessian = hf(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:

        x = x - lr * np.dot(np.linalg.inv(hessian), gradient)
        lr *= lr_decr

        points.append(x)
        nit += 1
        gradient = gf(x)
        hessian = hf(x)

    return x, points, nit


def newton_ls(x: np.array, f: Callable, gf: Callable, hf: Callable, maxiter=100, tol=0.001,
              method='brent') -> Tuple[np.ndarray, List[np.array], int]:
    """
    Aplica el método de Newton para encontrar el mínimo de una función cuyo argumento es
    multidimensional, usando además sondeo lineal para hallar el valor óptimo del factor
    de aprendizaje (`lr_opt`) para cada iteración.

    Parameters:
    ----------
        - `x` (np.array): Un array que representa el punto inicial donde empieza el algoritmo.
        - `f` (Callable): La función a minimizar.
        - `gf` (Callable): La función que calcula el gradiente de 'f'.
        - `hf` (Callable): La función que calcula la matriz hessiana de 'f'.
        - `maxiter` (int, opcional): El número máximo de iteraciones. Por defecto, es 100.
        - `tol` (float, opcional): La tolerancia para detener el algoritmo. Por defecto, es 0.001.
        - `method` (str, opcional): El método para encontrar la tasa de aprendizaje. Por defecto,
        es 'brent'.

    Returns:
    --------
        - `Tuple[np.ndarray, List[np.array], int]`:
            - El punto aproximado al mínimo.
            - Una lista de puntos intermedios (arrays) calculados durante la optimización.
            - El número de iteraciones realizadas.

    """
    points = [x]
    nit = 0
    gradient = gf(x)
    hessian = hf(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:

        direction = np.dot(np.linalg.inv(hessian), gradient)
        f_1d = f_md(f, x, direction)

        res = minimize_scalar(f_1d, method=method)
        lr_opt = res.x

        x = x - lr_opt * direction

        points.append(x)
        nit += 1
        gradient = gf(x)
        hessian = hf(x)

    return x, points, nit


def nesterov_grad_nd(x: np.array, f: Callable, df: Callable, tol: float = 1e-6, lr: float = 0.1,
                    beta: float = 0.9, maxiter: int=100) -> Tuple[np.array, List[np.array], int]:
    """
    Implementación del algoritmo de Nesterov del Gradiente Acelerado en múltiples dimensiones.

    Parameters:
    -----------
        - `x` (np.array): Punto inicial en el espacio de varias dimensiones.
        - `f` (Callable): Función objetivo que se desea minimizar.
        - `df`(Callable): Función que calcula el gradiente de la función objetivo.
        - `tol` (float, optional): Tolerancia para la convergencia del gradiente (predeterminado: 1e-6).
        - `lr` (float, optional): Tasa de aprendizaje (predeterminado: 0.1).
        - `beta` (float, optional): Parámetro de aceleración de Nesterov (predeterminado: 0.9).
        - `maxiter` (int, optional): Máximo de iteraciones permitidas por el algoritmo. Por defecto es 100.

    Returns:
    --------
        - Tuple[np.array, List[np.array], int]:
            - `x` : El punto que aproxima el mínimo local de la función.
            - `nit`: Número de iteraciones realizadas.
            - `nfev`: Número total de evaluaciones de la función objetivo y su gradiente.
    
    """
    points = [x]
    x_a = x
    y = x
    nit, nfev = 0, 2

    while nit < maxiter and np.linalg.norm(df(x)) > tol:
        y = x + beta * (x - x_a)
        x = y - lr * df(y)
        x_a = x
        points.append(x)
        nit += 1
        nfev += 2

    return x, points, nit, nfev


def _UpdateD(D: np.ndarray, gf: Callable, y: np.ndarray, ly: np.ndarray, tol:float) -> np.ndarray:
        """
        Update the inverse Hessian matrix approximation (D) in the DFP algorithms.

        Parameters:
        -----------
            - `D` (np.ndarray): Current inverse Hessian matrix approximation.
            - `gf` (Callable): Gradient (first-order derivative) of the objective function.
            - `y` (np.ndarray): Current point in the optimization process.
            - `ly` (np.ndarray): Previous point in the optimization process.
            - `tol` (float): Tolerance for numerical stability.

        Returns:
        --------
            `np.ndarray`: Updated inverse Hessian matrix approximation.
        """
        p = ly - y
        q = gf(ly) - gf(y)

        p_trans = p.T
        q_trans = q.T
        return D + np.outer(p, p_trans) / (np.dot(p_trans, q) + tol) - np.outer(np.dot(D, q), np.dot(q_trans, D)) / (np.dot(np.dot(q_trans, D), q) + tol)


def dfp(x: np.array, f: Callable, gf: Callable, lr = 0.01, maxiter=100, tol=0.001) -> Tuple[np.ndarray, List[np.array], int]:
    """
    Minimize a scalar function of one or more variables using the `DFP` (Davidon-Fletcher-Powell) method.
    This method approximates the inverse Hessian matrix (similar to `newton`method) and updates it in each iteration.

    Parameters:
    -----------
        - `x` (np.array): Initial guess for the minimum point.
        - `f` (Callable): Objective function to be minimized.
        - `gf` (Callable): Gradient (first-order derivative) of the objective function.
        - `lr` (float, optional):  Learning rate or step size for each iteration. Default is 0.01.
        - `maxiter` (int, optional): Maximum number of iterations. Default is 100.
        - `tol` (float, optional): Tolerance to declare convergence. Default is 0.001.

    Returns:
    --------
    - `Tuple[np.ndarray, List[np.array], int]`
        - `x` (np.ndarray): The point where the algorithm stopped.
        - `points` (List[np.array]): List of points visited during optimization.
        - `nit` (int): Number of iterations performed.
    """
    points = []
    nit = 0
    gradient = gf(x)
    n = len(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:
        y = x
        D = np.identity(n)

        for _ in range(n):
            direction = - np.dot(D, gf(y))
            ly = y
            y = y + lr * direction
            D = _UpdateD(D,gf,y,ly,tol)

        x = y
        gradient = gf(x)
        points.append(x)
        nit += 1
    return x, points, nit


def dfp_ls(x: np.array, f: Callable, gf: Callable, maxiter=100, tol=0.001, method='brent') -> Tuple[np.ndarray, List[np.array], int]:
    """
    Minimize a scalar function of one or more variables using the `DFP` (Davidon-Fletcher-Powell) method and using linear search
    to found the new `lr` value in each iteration.

   This method approximates the inverse Hessian matrix (similar to `newton`method) and updates it in each iteration.

    Parameters:
    -----------
        - `x` (np.array): Initial guess for the minimum point.
        - `f` (Callable): Objective function to be minimized.
        - `gf` (Callable): Gradient (first-order derivative) of the objective function.
        - `lr` (float, optional):  Learning rate or step size for each iteration. Default is 0.01.
        - `maxiter` (int, optional): Maximum number of iterations. Default is 100.
        - `tol` (float, optional): Tolerance to declare convergence. Default is 0.001.

    Returns:
    --------
    - `Tuple[np.ndarray, List[np.array], int]`
        - `x` (np.ndarray): The point where the algorithm stopped.
        - `points` (List[np.array]): List of points visited during optimization.
        - `nit` (int): Number of iterations performed.
    """
    points = []
    nit = 0
    gradient = gf(x)
    n = len(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:
        y = x
        D = np.identity(n)

        for _ in range(n):
            direction = -np.dot(D, gf(y))
            f_1d = f_md(f, y, -direction)

            res = minimize_scalar(f_1d, method=method)
            lr_opt = res.x

            ly = y
            y = y + lr_opt * direction
            D = UpdateD(D,gf,y,ly,tol)
        x = y
        gradient = gf(x)
        points.append(x)
        nit += 1
    return x, points, nit


def fr(x: np.array, f: Callable, gf: Callable, lr = 0.01, maxiter=100, tol=0.001) -> Tuple[np.ndarray, List[np.array], int]:
    """
    Minimize a scalar function of one or more variables using the FR (Fletcher-Reeves) method.

    Parameters:
    -----------
        - `x` (np.array): Initial guess for the minimum point.
        - `f` (Callable): Objective function to be minimized.
        - `gf` (Callable): Gradient (first-order derivative) of the objective function.
        - `lr` (float, optional):  Learning rate or step size for each iteration. Default is 0.01.
        - `maxiter` (int, optional): Maximum number of iterations. Default is 100.
        - `tol` (float, optional): Tolerance to declare convergence. Default is 0.001.

    Returns:
    --------
    - `Tuple[np.ndarray, List[np.array], int]`
        - `x` (np.ndarray): The point where the algorithm stopped.
        - `points` (List[np.array]): List of points visited during optimization.
        - `nit` (int): Number of iterations performed.
    """
    points = []
    nit = 0
    n = len(x)
    gradient = gf(x)

    
    while nit < maxiter and np.linalg.norm(gradient) > tol:
        y = x
        direction = - gradient
        for _ in range(n):
            ly = y
            y = y + lr * direction
            
            gy = gf(y)
            a = (np.linalg.norm(gy)**2)/(np.linalg.norm(gf(ly))**2)
            direction = - gy + a * direction
            
        x = y
        gradient = gf(x)
        points.append(x)
        nit += 1
    return x, points, nit


def fr_ls(x: np.array, f: Callable, gf: Callable, maxiter=100, tol=0.001, method='brent') -> Tuple[np.ndarray, List[np.array], int]:
    """
    Minimize a scalar function of one or more variables using the FR (Fletcher-Reeves) method and using linear search
    to found the new `lr` value in each iteration.

    Parameters:
    -----------
        - `x` (np.array): Initial guess for the minimum point.
        - `f` (Callable): Objective function to be minimized.
        - `gf` (Callable): Gradient (first-order derivative) of the objective function.
        - `lr` (float, optional):  Learning rate or step size for each iteration. Default is 0.01.
        - `maxiter` (int, optional): Maximum number of iterations. Default is 100.
        - `tol` (float, optional): Tolerance to declare convergence. Default is 0.001.

    Returns:
    --------
    - `Tuple[np.ndarray, List[np.array], int]`
        - `x` (np.ndarray): The point where the algorithm stopped.
        - `points` (List[np.array]): List of points visited during optimization.
        - `nit` (int): Number of iterations performed.
    """
    points = []
    nit = 0
    n = len(x)
    gradient = gf(x)

    while nit < maxiter and np.linalg.norm(gradient) > tol:
        y = x
        direction = -gradient
        for _ in range(n):
            f_1d = f_md(f, y, -direction)

            res = minimize_scalar(f_1d, method=method)
            lr_opt = res.x

            ly = y
            y = y + lr_opt * direction
  
            gy = gf(y)
            a = np.linalg.norm(gy)**2/np.linalg.norm(gf(ly) + tol)**2 # Sumamos tol para evitar divisiones entre 0
            direction = - gy + a * direction

        x = y
        gradient = gf(x)
        points.append(x)
        nit += 1
    return x, points, nit


def f_rosen_md(z: np.ndarray, a=1, b=100) -> float:
    """
    Calcula la función (no convexa) de Rosenbrock multidimensional.

    Parameters:
    ----------
        - `z` (np.ndarray): Un punto (array) en el que se evalúa la función.
        - `a` (float, optional): Parámetro a. Por defecto es 1.
        - `b` (float, optional): Parámetro b. Por defecto es 100.

    Returns:
    --------
        `float`: El valor de la función de Rosenbrock en el punto.

    References:
    -----------
        https://en.wikipedia.org/wiki/Rosenbrock_function.
    """
    return np.sum((a - z[:-1]) ** 2 + b * (z[1:] - z[:-1] ** 2) ** 2)


def save_x(x: np.ndarray) -> None:
    """
    Callback para guardar puntos intermedios durante la optimización en la lista global l_x.

    Parameters:
    ----------
        - `x` (np.ndarray): Punto intermedio hallado durante la optimización.

    Returns:
    --------
        `None`
    """
    l_x.append(x)


def my_minimize(fun: Callable, x0: np.ndarray, method='BFGS', tol=1e-4,
                callback=save_x) -> scipy.optimize._optimize.OptimizeResult:
    """
    Realiza la optimización de la función dada utilizando el método especificado. Se utiliza
    el método jacobian de autograd para hallar el jacobiano de la función pasada como argumento.
    Además, se inicializa una lista vacía de puntos intermedios que se guardarán durante la
    optimización.

    Parameters:
    ----------
        - `fun` (Callable): La función que se va a minimizar.
        - `x0` (np.ndarray): El punto inicial para la optimización.
        - `method` (str): El método de optimización a utilizar (por defecto, 'BFGS').
        - `tol` (float): La tolerancia para la convergencia (por defecto, 1e-4).
        - `callback` (Callable): Función de devolución de llamada para puntos intermedios
        (por defecto, `save_x`).

    Returns:
    --------
        `scipy.optimize._optimize.OptimizeResult`: El objeto del resultado de la optimización
        con variedad de información.
    """
    global l_x
    l_x = []

    if method in ['BFGS', 'L-BFGS-B', 'CG']:
        jacobiano = jacobian(fun)

    else:
        jacobiano = None

    return minimize(fun=fun, x0=x0, jac=jacobiano, method=method, tol=tol,
                    callback=callback, options={'gtol': tol})


def get_l_x() -> List:
    """
    Función auxiliar que devuelve la lista de puntos intermedios guardados durante la optimización.

    Parameters:
    ----------
        `None`

    Returns:
    --------
        `List`: Lista de puntos intermedios calculados durante la optimización.
    """
    return l_x
