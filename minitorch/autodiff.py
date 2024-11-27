from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol

from collections import defaultdict


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    f_plus = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
    f_minus = f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """For storing the derivative of the variable"""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for the variable"""
        ...

    def is_leaf(self) -> bool:
        """Whether the variable is a leaf node"""
        ...

    def is_constant(self) -> bool:
        """Whether the variable is a constant node"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Iterable of parent variables"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Chain rule for the variable"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    order = []

    def dfs(var: Variable) -> None:
        # corrected from if var in visited or var.is_constant():
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not var.is_constant():
                    dfs(parent)
        visited.add(var.unique_id)
        order.append(var)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable
    deriv: The derivative of the right-most variable to propagate backward

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    computation_graph = topological_sort(variable)

    scalar_to_derivative = defaultdict(float)
    scalar_to_derivative[variable.unique_id] = deriv

    for var in computation_graph:
        if var.is_leaf():
            var.accumulate_derivative(scalar_to_derivative[var.unique_id])
        else:
            for parent, deriv in var.chain_rule(scalar_to_derivative[var.unique_id]):
                if parent.is_constant():
                    continue
                scalar_to_derivative[parent.unique_id] += deriv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the values saved for backward computation."""
        return self.saved_values
