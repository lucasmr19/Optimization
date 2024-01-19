# Algoritmos 1D en Python

Este archivo de código contiene implementaciones en Python de algoritmos para encontrar raíces y mínimos de funciones unidimensionales. Los algoritmos implementados incluyen:

1. **Método de la Bisección:**
   - Función `bisection(f, a, b, tol=0.001, maxiter=100)`.
   - Encuentra una raíz de la función \(f\) dentro del intervalo \([a, b]\) utilizando el método de la bisección.

2. **Método de la Falsa Posición:**
   - Función `regula_falsi(f, a, b, tol=0.001, maxiter=100)`.
   - Encuentra una raíz de la función \(f\) dentro del intervalo \([a, b]\) utilizando el método de la falsa posición.

3. **Método de la Secante:**
   - Función `secant(f, a, b, tol=0.001, maxiter=100)`.
   - Encuentra una raíz de la función \(f\) dentro del intervalo \([a, b]\) utilizando el método de la secante.

4. **Método de la Trisección:**
   - Función `trisection(f, bracket, xtol=0.001, maxiter=100)`.
   - Encuentra un mínimo de la función \(f\) dentro del bracket dado utilizando el método de trisección.

5. **Método de la Razón Áurea (Golden Section):**
   - Función `golden(f, bracket, xtol=0.001, maxiter=100)`.
   - Encuentra un mínimo de la función \(f\) dentro del bracket dado utilizando el método de la razón áurea.

6. **Funciones de Utilidad:**
   - `root_bracket(f, a, b, delta=1.)`: Encuentra un intervalo donde la función \(f\) cambia de signo.
   - `min_bracket(f, a, b, delta=1.0)`: Encuentra un bracket para el mínimo de la función \(f\).

### Uso
```python
import numpy as np
from algorithms_1d import *

# Definir la función
def f(x):
    return x**2 - 4

# Ejemplo de uso del método de la bisección
a, b = root_bracket(f, -3, 3)
root, iterations = bisection(f, a, b)
print(f"Raíz encontrada: {root} después de {iterations} iteraciones.")
```

### Excepciones
- `IterationError`: Se lanza si se excede el número máximo de iteraciones.
- `ValueError`: Se lanza si no se encuentra una raíz en el intervalo dado.

**Nota:** Asegúrate de tener instalada la biblioteca NumPy antes de ejecutar el código:

```bash
pip install numpy
```
