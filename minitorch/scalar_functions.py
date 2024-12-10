from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply a scalar function to a set of values, and save the computation graph.

        This method will create a new computation graph, and then apply the
        forward and backward functions.

        Parameters
        ----------
        *vals : ScalarLike
            The values of the function to apply.

        Returns
        -------
        Scalar
            The value of the function applied to the inputs.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(scalars[-1].data)

        # Create the context
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward computation of the Add function.

        Parameters
        ----------
        ctx : Context
            The computation context.
        a : float
            The first input value.
        b : float
            The second input value.

        Returns
        -------
        float
            The sum of the two input values.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward computation of the Add function.

        Parameters
        ----------
        ctx : Context
            The computation context.
        d_output : float
            The output derivative.

        Returns
        -------
        Tuple[float, ...]
            The derivatives of the output with respect to the inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward computation of the Log function.

        Parameters
        ----------
        ctx : Context
            The computation context.
        a : float
            The input value.

        Returns
        -------
        float
            The log of the input value.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward computation of the Log function.

        Parameters
        ----------
        ctx : Context
            The computation context.
        d_output : float
            The output derivative.

        Returns
        -------
        float
            The derivative of the output with respect to the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward computation of the Mul function.

        Parameters
        ----------
        ctx : Context
            The computation context.
        x : float
            The first input value.
        y : float
            The second input value.

        Returns
        -------
        float
            The multiplication of the two input values.

        """
        ctx.save_for_backward(x, y)
        return operators.mul(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """The derivative of the Mul function with respect to its inputs.

        Parameters
        ----------
        ctx : Context
            The computation context.
        d_output : float
            The output derivative.

        Returns
        -------
        Tuple[float, ...]
            The derivative of the output with respect to the input.

        """
        (x, y) = ctx.saved_values
        return (d_output * y, d_output * x)


class Inv(ScalarFunction):
    """Neg function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for inverse function."""
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse function."""
        (x,) = ctx.saved_values
        return operators.inv_back(x, d_output)


class Neg(ScalarFunction):
    """Neg function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for negation function."""
        return operators.neg(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation function."""
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1/(1+e^-x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for sigmoid function."""
        ctx.save_for_backward(x)
        return operators.sigmoid(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function."""
        (x,) = ctx.saved_values
        return (operators.sigmoid(x) ** 2) * operators.exp(operators.neg(x)) * d_output


class ReLU(ScalarFunction):
    """RELU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for ReLU function."""
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function."""
        (x,) = ctx.saved_values
        return operators.relu_back(x, d_output)


class Exp(ScalarFunction):
    """Exp function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Forward pass for exponential function."""
        ctx.save_for_backward(x)
        return operators.exp(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential function."""
        (x,) = ctx.saved_values
        return operators.exp(x) * d_output


class LT(ScalarFunction):
    """Less than function"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass for less than function."""
        return operators.lt(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than function."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function"""

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Forward pass for equal function."""
        return operators.eq(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equal function."""
        return 0.0, 0.0
