from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Jit compile a function for fast CPU execution."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # similar implementatino to standard tensor_map except with
        # prange instead of range
        # Main loop in parallel
        # All indices use numpy buffers
        if len(out_shape) > MAX_DIMS or len(in_shape) > MAX_DIMS:
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")

        # When out and in are stride-aligned, avoid indexing
        if (
            len(out_shape) == len(in_shape)
            and len(out_strides) == len(in_strides)
            and (out_shape == in_shape).all()
            and (out_strides == in_strides).all()
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])

        else:
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
                in_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
                # Convert flat index to multidimensional index
                # to_index function is useful just for getting
                # continous set of indices
                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, in_shape, in_index)

                # ordinals
                in_position = index_to_position(in_index, in_strides)
                out_position = index_to_position(out_index, out_strides)

                out[out_position] = fn(in_storage[in_position])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # similar implementatino to standard tensor_zip except with
        # prange instead of range
        if (
            len(out_shape) > MAX_DIMS
            or len(a_shape) > MAX_DIMS
            or len(b_shape) > MAX_DIMS
        ):
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")

        n = len(out)
        if (
            len(out_strides) == len(a_strides)
            and len(out_shape) == len(a_shape)
            and len(a_strides) == len(b_strides)
            and len(b_shape) == len(a_shape)
            and (out_strides == a_strides).all()
            and (a_strides == b_strides).all()
            and (out_shape == a_shape).all()
            and (a_shape == b_shape).all()
        ):
            for i in prange(n):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # global broacasting case
            for i in prange(n):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                in_index_a = np.empty(MAX_DIMS, dtype=np.int32)
                in_index_b = np.empty(MAX_DIMS, dtype=np.int32)
                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, a_shape, in_index_a)
                broadcast_index(out_index, out_shape, b_shape, in_index_b)

                # Calculate ordinals
                in_position_a = index_to_position(in_index_a, a_strides)
                in_position_b = index_to_position(in_index_b, b_strides)
                out_position = index_to_position(out_index, out_strides)

                out[out_position] = fn(
                    a_storage[in_position_a], b_storage[in_position_b]
                )

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # similar implementatino to standard tensor_reduce except with
        # prange instead of range
        if len(out_shape) > MAX_DIMS or len(a_shape) > MAX_DIMS:
            raise ValueError(f"Tensor dimensions cannot exceed {MAX_DIMS}")

        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, dtype=np.int32)
            reduce_size: int = a_shape[reduce_dim]
            to_index(i, out_shape, out_index)
            output_position = index_to_position(out_index, out_strides)

            reduce_stride = a_strides[reduce_dim]
            baseIdx = index_to_position(out_index, a_strides)
            acc = out[output_position]
            for j in range(reduce_size):
                a_position = baseIdx + j * reduce_stride
                acc = fn(acc, float(a_storage[a_position]))

            out[output_position] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    assert a_shape[-1] == b_shape[-2], "a_shape[-1] != b_shape[-2]"
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0
    for batch in prange(out_shape[0]):
        batch_idx_a = batch * a_batch_stride
        batch_idx_b = batch * b_batch_stride
        out_batch_idx = batch * out_batch_stride
        for r in range(out_shape[-2]):
            for c in range(out_shape[-1]):
                acc = 0.0
                for i in range(a_shape[-1]):  # want to do the inner loop before col
                    # like BLAS implementation to be more cache friendly,
                    # but this causes global writes
                    acc += (
                        a_storage[batch_idx_a + r * a_strides[-2] + i * a_strides[-1]]
                        * b_storage[batch_idx_b + i * b_strides[-2] + c * b_strides[-1]]
                    )
                out[out_batch_idx + r * out_strides[-2] + c * out_strides[-1]] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
