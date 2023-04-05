
import numpy as np

from typing import Callable


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def sub(a, b):
    return b - a


def remap(v, x, y, clip=False):
    if x[1] == x[0]:
        return y[0]
    out = y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])
    if clip:
        out = constrain(out, y[0], y[1])
    return out


def bernoulli_kullback_leibler(p: float, q: float) -> float:
    """Compute the Kullback-Leibler divergence of 
    two Bernoulli distributions.

    Parameters
    ----------
    p : float
        Parameter of the first Bernoulli distribution.
    q : float
        Parameter of the second Bernoulli distribution.

    Returns
    -------
    float
        KL(B(p) || B(q))
    """
    kl1, kl2 = 0, np.infty
    if p > 0:
        if q > 0:
            kl1 = p*np.log(p/q)

    if q < 1:
        if p < 1:
            kl2 = (1 - p) * np.log((1 - p) / (1 - q))
        else:
            kl2 = 0
    return kl1 + kl2


def d_bernoulli_kullback_leibler_dq(p: float, q: float) -> float:
    """Compute the partial derivative of the Kullback-Leibler 
    divergence of two Bernoulli distributions. With respect to 
    the parameter q of the second distribution.

    Parameters
    ----------
    p : float
        Parameter of the first Bernoulli distribution.
    q : float
        Parameter of the second Bernoulli distribution.

    Returns
    -------
    float
        dKL/dq(B(p) || B(q))
    """
    return (1 - p) / (1 - q) - p/q


def kl_upper_bound(_sum: float, count: int, threshold: float = 1, eps: float = 1e-2, lower: bool = False) -> float:
    """Upper Confidence Bound of the empirical mean built 
    on the Kullback-Leibler divergence. The computation 
    involves solving a small convex optimization problem 
    using Newton Iteration.

    Parameters
    ----------
    _sum : float
        Sum of sample values.
    count : int
        Number of samples.
    threshold : float, optional
        The maximum `kl-divergence * count`, by default 1
    eps : float, optional
        Absolute accuracy of the Newton Iteration, by default 1e-2
    lower : bool, optional
        Whether to compute a lower-bound instead of upper-bound, by default False

    Returns
    -------
    float
        The Upper Confidence Bound.
    """
    if count == 0:
        return 0 if lower else 1

    mu = _sum/count
    max_div = threshold/count

    # Solve KL(mu, q) = max_div
    def kl(q): return bernoulli_kullback_leibler(mu, q) - max_div
    def d_kl(q): return d_bernoulli_kullback_leibler_dq(mu, q)
    a, b = (0, mu) if lower else (mu, 1)

    return newton_iteration(kl, d_kl, eps, a=a, b=b)


def newton_iteration(f: Callable, df: Callable, eps: float, x0: float = None, a: float = None, b: float = None,
                     weight: float = 0.9, display: bool = False, max_iterations: int = 100) -> float:
    """Run Newton Iteration to solve $$f(x) = 0, \text{ with } x \text{ in } [a, b]$$.

    Parameters
    ----------
    f : Callable
        A function `R -> R`.
    df : Callable
        The function derivative.
    eps : float
        The desired accuracy.
    x0 : float, optional
        An initial value, by default None
    a : float, optional
        An optional lower-bound, by default None
    b : float, optional
        An optional upper-bound, by default None
    weight : float, optional
        A weight to handle out of bounds events, by default 0.9
    display : bool, optional
        Plot the function, by default False
    max_iterations : int, optional
        Iteration threshold, by default 100

    Returns
    -------
    float
        $x$ such that $f(x) = 0$
    """
    x = np.inf
    if x0 is None:
        x0 = (a + b) / 2
    if a is not None and b is not None and a == b:
        return a
    x_next = x0
    iterations = 0
    while abs(x - x_next) > eps and iterations < max_iterations:
        iterations += 1
        x = x_next

        if display:
            import matplotlib.pyplot as plt
            xx0 = a or x-1
            xx1 = b or x+1
            xx = np.linspace(xx0, xx1, 100)
            yy = np.array(list(map(f, xx)))
            plt.plot(xx, yy)
            plt.axvline(x=x)
            plt.show()

        f_x = f(x)
        try:
            df_x = df(x)
        except ZeroDivisionError:
            df_x = (f_x - f(x-eps))/eps
        if df_x != 0:
            x_next = x - f_x / df_x

        if a is not None and x_next < a:
            x_next = weight * a + (1 - weight) * x
        elif b is not None and x_next > b:
            x_next = weight * b + (1 - weight) * x

    if a is not None and x_next < a:
        x_next = a
    if b is not None and x_next > b:
        x_next = b

    return x_next


def all_argmax(x: np.ndarray) -> np.ndarray:
    """Returns the non-zero elements of a np.ndarray like 
    structure which are the row-wise maximum values of that
    structure.

    Parameters
    ----------
    x : np.ndarray
        A set.

    Returns
    -------
    np.ndarray
        The list of indexes of all maximums of `x`.
    """
    m = np.amax(x)
    return np.nonzero(np.isclose(x, m))[0]


def random_argmax(x: np.ndarray) -> int:
    """Randomly tie-breaking $argmax$.

    Parameters
    ----------
    x : np.ndarray
        An array.

    Returns
    -------
    int
        A random index among the maximums.
    """
    indices = all_argmax(x)
    return np.random.choice(indices)
