"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, TypeVar

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$
epsilon = 1e-2


def mul(x: float, y: float) -> float:
    """Multiply two numbers element-wise."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers element-wise."""
    return x + y


def neg(x: float) -> float:
    """Negate a number element-wise."""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y element-wise."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y element-wise."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Check if x is greater than y element-wise."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close (exponentially) element-wise."""
    return abs(x - y) < 1e-8


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function of a value."""
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    """Calculate the relu function of a value."""
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Calculate the log function of a value."""
    return math.log(x)


def exp(x: float) -> float:
    """Calculate the exp function of a value."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the inverse function of a value."""
    if -epsilon < x < epsilon:  # Close to zero
        x = epsilon if x >= 0 else -epsilon
    return 1.0 / x


def log_back(x: float, grad_output: float) -> float:
    """Calculate the log backward function of a value times grad."""
    return grad_output / (x + epsilon)


def inv_back(x: float, grad_output: float) -> float:
    """Calculate the inverse backward function of a value times grad."""
    # return -1 * inv(x) * inv(x) * grad_output
    return -(1.0 / (x * x)) * grad_output


def relu_back(x: float, grad_output: float) -> float:
    """Calculate the relu backward function of a value times grad."""
    return grad_output if x > 0.0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def map(fn: Callable[[A], B], items: Iterable[A]) -> Iterable[B]:
    """Higher-order function that applies a given function to each element of an iterable"""
    if not items:
        return []
    return [fn(item) for item in items]


def zipWith(
    fn: Callable[[A, B], C], items1: Iterable[A], items2: Iterable[B]
) -> Iterable[C]:
    """Higher-order function that combines elements from two iterables using a given function"""
    results = []
    items1 = iter(items1)
    items2 = iter(items2)
    while True:
        try:
            items1_i = next(items1)
            items2_i = next(items2)

            results.append(fn(items1_i, items2_i))
        except StopIteration:
            # Iterable is completely consumed
            break

    return results


def reduce(fn: Callable[[A, A], A], items: Iterable[A], init: A) -> A:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    value = init
    for item in items:
        value = fn(value, item)
    return value


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use :func:`map` and :func:`neg` to negate each element in a list."""
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Use :func:`zipWith` and :func:`add` to add corresponding elements of lists."""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Use :func:`reduce` and :func:`add` to sum up a list of numbers."""
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Use :func:`reduce` and :func:`mul` to multiply the elements of a list."""
    return reduce(mul, ls, 1.0)
