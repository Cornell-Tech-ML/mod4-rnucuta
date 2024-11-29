"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Used for initializing tensors that require context tracking"""
        self.history = History()

    def requires_grad(self) -> bool:
        """Used for checking if history tracking is on"""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self._tensor.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create new complete Tensor object with stride"""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if nothing has been applied to tensor"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Analagous to scalar parents operation"""
        assert self.history is not None
        return self.history.inputs

    @property
    def size(self) -> int:
        """Returns size of tensor array"""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns number of dims of tensor array"""
        return self._tensor.dims

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule to tensors"""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(
            h.inputs
        ), f"Bug in function {h.last_fn}, {len(x)}, {len(h.inputs)} {h.inputs}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            grad_output (Tensor, opt): starting derivative to backpropagate through the model
                                       (typically left out initially, and assumed to be 1.0).

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Divides a by b"""
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Divides B by A"""
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    # def __hash__(self) -> int:
    #     return id(self)

    def __add__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, self._ensure_tensor(b))

    def __radd__(self, b: TensorLike) -> Tensor:
        return self + b
        # return  Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        return Add.apply(self, -self._ensure_tensor(b))

    def __rsub__(self, b: TensorLike) -> Tensor:
        return self - b

    def __mul__(self, b: TensorLike) -> Tensor:
        # if isinstance(b, Tensor):
        #     return Mul.apply(self, b)
        return Mul.apply(self, self._ensure_tensor(b))

    def __rmul__(self, b: TensorLike) -> Tensor:
        return self * b

    def __lt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        return Neg.apply(self)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Returns True if all elements in the tensor are True, False otherwise.
        If dim is specified, perform the operation along that dimension.
        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))
        # return All.apply(
        #     self,
        #     self._ensure_tensor(dim) if dim is not None else self._ensure_tensor(-1),
        # )

    def is_close(self, b: Tensor) -> Tensor:
        """Returns tensor indicating elementwise if a and b are close."""
        return IsClose.apply(self, b)

    def sigmoid(self) -> Tensor:
        """Returns sigmoid of self"""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Returns relu of self"""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Returns log of self"""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Returns log of self"""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Returns sum of self"""
        if dim is None:  # Sum over all dimensions
            # return Sum.apply(self, self._ensure_tensor(-1))
            # total = self
            # for d in range(self.dims):
            #     total = Sum.apply(total, self._ensure_tensor(d))
            # return total.view(1)
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Takes mean of self. Dimension is optional"""
        # if dim is not None:
        #     dim = self._ensure_tensor(dim)
        # axis_total = (
        #     self._tensor.size
        #     if dim is None
        #     else self.shape[int(self._ensure_tensor(dim).item())]
        # )
        # return self.sum(dim) / axis_total
        if dim is not None:
            return self.sum(dim) / float(self.shape[dim])
        else:
            return self.sum() / float(self.size)

    def view(self, *shape: int) -> Tensor:
        """Changes view of tensor. Unsure how this is supposed to take dim optionally"""
        return View.apply(self, tensor(list(shape)))

    def permute(self, *order: int) -> Tensor:
        """Changes permutation of tensor's shape"""
        return Permute.apply(self, tensor(list(order)))

    def zero_grad_(self) -> None:
        """Sets grad to None"""
        self.grad = None
