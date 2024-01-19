"""
This module contains basic Python implementations of optimization algorithms for finding roots and minima of one-dimensional functions.

Author: 
%  ___                                                       _     __      
% /\_ \                                                    /' \  /'_ `\    
% \//\ \   __  __   ___     __      ____   ___ ___   _ __ /\_, \/\ \L\ \   
%   \ \ \ /\ \/\ \ /'___\ /'__`\   /',__\/' __` __`\/\`'__\/_/\ \ \___, \  
%    \_\ \\ \ \_\ /\ \__//\ \L\.\_/\__, `/\ \/\ \/\ \ \ \/   \ \ \/__,/\ \ 
%    /\____\ \____\ \____\ \__/.\_\/\____\ \_\ \_\ \_\ \_\    \ \_\   \ \_\
%    \/____/\/___/ \/____/\/__/\/_/\/___/ \/_/\/_/\/_/\/_/     \/_/    \/_/
"""

from typing import Callable, Tuple
import numpy as np


class IterationError(Exception):
    """
    Custom exception to indicate that the maximum number of iterations allowed has been exceeded.

    This exception is used to handle situations where an algorithm or function has reached
    a maximum number of iterations, typically the parameter 'maxiter = 100', without finding the desired result.
    """
    pass

def root_bracket(f: Callable, a: float, b: float, delta=.1)-> Tuple[float,float]:
    """
    Finds an interval [a, x] where the function f changes sign, implying that at least
    one root of the function f passed as an argument is present.

    Parameters:
    -----------
        - `f` (Callable): The function whose root is to be found.
        - `a` (float): The initial point of the interval.
        - `b` (float): The final point of the interval.
        - `delta` (float): The step to explore the interval. Defaults to .1.

    Returns:
    --------
        `Tuple[float,float]`: A tuple containing the endpoints of the interval [a, x] where f changes sign.
        If no such interval is found, the interval (-∞, ∞) is returned.

    Raises:
    -------
        `ValueError`: Raised if no root is found in the given interval.
    """
    x = a + delta
    while x < b:
        if f(a)*f(x) < 0:
            return a,x
        else:
            x += delta
    raise ValueError('No root exists in the given interval')


def bisection(f: Callable, a: float, b: float, tol=1e-6, maxiter=100)-> Tuple[float, int]:
    """
    Finds the roots of a function using the bisection method.

    Parameters:
    -----------
        - `f` (Callable): The function whose root is to be found.
        - `a` (float): The initial point of the interval.
        - `b` (float): The final point of the interval.
        - `tol` (float): Tolerance. Defaults to 1e-6.
        - `maxiter` (int): The maximum number of iterations. Defaults to 100.

    Returns:
    --------
        - `Tuple[float, int]`: A tuple containing the approximate root and the number of iterations performed.

    Raises:
    -------
        - `IterationError`: Raised if the maximum number of iterations is exceeded.
        - `ValueError`: Raised if no root is found in the given interval.
    """
    if f(a) * f(b) > 0:
        raise ValueError('No root exists in the given interval')
    
    nit = 0

    while nit <= maxiter: 
        nit += 1           
        c = (a + b) / 2 # Punto medio

        if abs(f(c)) <= tol:
            return c, nit
        
        elif f(a) * f(c) < 0:
            b = c

        else:
            a = c

    raise IterationError('The maximum number of allowed iterations has been exceeded.')


def regula_falsi(f: Callable, a: float, b: float, tol=1e-6, maxiter=100) -> Tuple[float, int]:
    """
    Finds the roots of a function using the false position method.

    Parameters:
    -----------
        - `f` (Callable): The function whose root is to be found.
        - `a` (float): The initial point of the interval.
        - `b` (float): The final point of the interval.
        - `tol` (float): Tolerance. Defaults to 1e-6.
        - `maxiter` (int): The maximum number of iterations. Defaults to 100.

    Returns:
    --------
        `Tuple[float,int]`: A tuple containing the approximate root and the number of iterations performed.
                  
    Raises:
    -------
        - `IterationError`: Raised if the maximum number of iterations is exceeded.
        - `ValueError`: No root exists in the given interval.
    """
    if f(b) * f(a) > 0:
        raise ValueError('No existe raiz en el intervalo dado')
    
    nit = 0

    while nit <= maxiter:
        nit += 1
        fa = f(a)
        fb = f(b)
        c = (a * fb - b * fa) / (fb - fa) # Paso clave
        
        if abs(f(c)) <= tol:
            return c, nit
        
        if f(c) < 0:
            a = c
        else:
            b = c
    
    raise IterationError("El número máximo de iteraciones permitidas ha sido excedido.")

def secant(f: Callable, a: float, b: float, tol=1e-6, maxiter=100)-> Tuple[float, int]:
    """
    Finds the roots of a function using the secant method.

    Parameters:
    -----------
        - `f` (Callable): The function whose root is to be found.
        - `a` (float): The initial point of the interval.
        - `b` (float): The final point of the interval.
        - `tol` (float): Tolerance. Defaults to 1e-6.
        - `maxiter` (int): The maximum number of iterations. Defaults to 100.

    Returns:
    --------
        `Tuple[float,int]`: A tuple containing the approximate root and the number of iterations performed.
                  
    Raises:
    -------
        - `IterationError`: Raised if the maximum number of iterations is exceeded.
        - `ValueError`: No root exists in the given interval.
    """
    if f(b) * f(a) > 0:
        raise ValueError('No existe raiz en el intervalo dado')
    
    nit = 0
    
    while nit <= maxiter:
        nit += 1
        fa = f(a)
        fb = f(b)        
        x_next = b - (fb * (b - a)) / (fb - fa) # Paso clave
        
        if abs(f(x_next)) <= tol:
            return x_next, nit
        
        a = b
        b = x_next
    raise IterationError("El número máximo de iteraciones permitidas ha sido excedido.")


def min_bracket(f: Callable, a: float, b: float, delta=.1) -> Tuple[float, float, float, int]:
    """Receives a function f, two floats a, b, and a step delta, and returns a tuple x0, x1, x2,
    nfev with three floats defining a bracket of f and an integer with the number of function evaluations
    performed.

    Parameters:
    -----------
        - `f` (Callable): Function to search for the bracket.
        - `a` (float): First float of the interval.
        - `b` (float): Second float of the interval.
        - `delta` (float): Step for the operation. Defaults to 0.1.

    Raises:
    -------
        `ValueError`: a cannot be greater than b.

    Returns:
    --------
        `Tuple[float, float, float, int]`: Returns the bracket (x0, x1, x2) and the number of function evaluations (nfev).
    """

    i = a
    j = b

    if a > b:
        raise ValueError('a no puede ser mayor que b')
    
    nfev = 0
    
    if f(a) < f(b):
        x0 = a
        x1 = a + delta
        while f(x1) < f(x0):
            x0 = x1
            x1 += delta
            nfev += 1
            if x1 >= b:
                return -np.inf, np.inf # No se encontró el bracket
    
    #f(a) > f(b)
    else:
        x0 = b
        x1 = b - delta
        while f(x1) < f(x0):
            x0 = x1
            x1 -= delta
            nfev += 1
            if x1 <= a:
                return -np.inf, np.inf # No se encontró el bracket
            
    if nfev == 0:
        return -np.inf, np.inf
    
 
    return i, x0, j, nfev




def trisection(f: Callable, bracket: Tuple[float, float, float], xtol=1e-6, maxiter=100) -> Tuple[float, int, int]:
    """
    Receives a function f, a tuple bracket defining a bracket of f, and, starting from that bracket, returns a tuple
    r, nit, nfev with an approximate minimum r and the numbers of iterations and function evaluations needed to find it
    using the trisection method.

    Parameters:
    -----------
        - `f` (Callable): Function to evaluate.
        - `bracket` (Tuple[float, float, float]): Bracket to search for the minimum.
        - `xtol` (float, optional): Minimum error margin. Defaults to 1e-6.
        - `maxiter` (int, optional): Maximum number of algorithm iterations. Defaults to 100.

    Raises:
    -------
        - `IterationError`: Exceeded the maximum number of iterations.
        - `ValueError`: The bracket does not meet the order conditions.

    Returns:
    --------
        `Tuple[float, int, int]`: Tuple with the root, the number of iterations, and the number of function evaluations.
    """
    a, b, c = bracket
    if not (a < b < c) and not (f(a) > f(b) and  f(b) < f(c)):
        raise ValueError('Incorrect bracket')

    
    nit = 0  # Número de iteraciones
    nfev = 0  # Número de evaluaciones de la función
    
    while abs(c - a) > xtol and nit <= maxiter:
        nit += 1
        nfev += 2  # Evaluamos la función en dos nuevos puntos
        
        # Calculamos los puntos intermedios
        x1 = a + (c - a) / 3
        x2 = c - (c - a) / 3
       
        f1 = f(x1)
        f2 = f(x2)
     
        if f1 < f2:
            c = x2
        else:
            a = x1
    
    if nit > maxiter:
        raise IterationError("El número máximo de iteraciones permitidas ha sido excedido.")
    
    r = (a + c) / 2
    return r, nit, nfev


def golden(f: Callable, bracket: Tuple[float, float, float], xtol=1e-6, maxiter=100) -> Tuple[float, int, int]:
    """
    Receives a function f, a tuple bracket defining a bracket of f, and starting from that bracket, returns a tuple
    r, nit, nfev with an approximate minimum r and the numbers of iterations and function evaluations needed to find it 
    using the golden ratio method.

    Parameters:
    -----------
        - `f` (Callable): Function to evaluate.
        - `bracket` (Tuple[float, float, float]): Bracket on which to search for the minimum.
        - `xtol` (float, optional): Minimum error margin. Defaults to 1e-6.
        - `maxiter` (int, optional): Maximum number of algorithm iterations. Defaults to 100.

    Raises:
    -------
        - `IterationError`: Exceeded the maximum number of iterations.
        - `ValueError`: The bracket does not meet the order conditions.

    Returns:
    --------
        `Tuple[float, int, int]`: Tuple with the root, the number of iterations, and the number of function evaluations.
    """
    a, b, c = bracket
    if not (a < b < c) and not (f(a) > f(b) and  f(b) < f(c)):
        raise ValueError('Incorrect bracket')
    
    nit = 0
    nfev = 2
    ra = 0.618033988749894
    x2 = a + (c - a) * ra
    x1 = c - (c - a) * ra
    f1 = f(x1)
    f2 = f(x2)
    
    while abs(c - a) > xtol and nit <= maxiter:
        nit += 1
        nfev += 1  

        if f1 < f2:
            c = x2  
            x2 = x1
            x1 = c - (c - a) * ra
            f2 = f1
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (c - a) * ra
            f1 = f2
            f2 = f(x2)

    if nit > maxiter:
        raise IterationError('The maximum number of allowed iterations has been exceeded.')
    
    r = (a + c) / 2
    return r, nit, nfev
