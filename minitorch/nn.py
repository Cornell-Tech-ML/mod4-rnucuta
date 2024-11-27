from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    tiled_tensor = input.contiguous().view(batch, channel, new_height, kh, new_width, kw).permute(0, 1, 2, 4, 3, 5).contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled_tensor, new_height, new_width

    # tiled_tensor = input.zeros((batch, channel, new_height, new_width, kh * kw))

    # # modified "view" on tensor
    # for b in range(batch):
    #     for c in range(channel):
    #         for h in range(new_height):
    #             for w in range(new_width):
    #                 # Fill the new tensor with the appropriate values from the input tensor
    #                 tiled_tensor[b, c, h, w] = input[b, c, h * kh:(h + 1) * kh, w * kw:(w + 1) * kw].reshape(-1)

    # return tiled_tensor, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width.

    """
    tiled_tensor, new_height, new_width = tile(input, kernel)
    avg_tensor = tiled_tensor.mean(dim=4)
    return avg_tensor.view(input.shape[0], input.shape[1], new_height, new_width)

max_reduce = FastOps.reduce(operators.max, float('-inf'))

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax along a specific dimension.

    Args:
    ----
        input: Tensor to compute argmax on.
        dim: Dimension along which to compute argmax.

    Returns:
    -------
        A 1-hot tensor with the same shape as `input` where the max indices are set to 1.
    
    """
    max_values = input.f.max_reduce(input, dim)
    return max_values == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int) -> Tensor:
        """Apply the max operator along a specific dimension.

        Args:
        ----
            ctx: Context for autograd.
            input: Tensor to compute max on.
            dim: Dimension along which to compute max.

        Returns:
        -------
            Tensor containing the max values along the specified dimension.

        """
        max_values = input.f.max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, dim)
        return max_values

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the max operation."""
        input, dim = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, argmax(input, dim)), 0.0


max = Max.apply

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specific dimension.

    Args:
    ----
        input: Tensor to compute softmax on.
        dim: Dimension along which to compute softmax.

    Returns:
    -------
        Tensor with softmax applied.

    """
    exponentiated = input.exp()
    return exponentiated / exponentiated.sum(dim)

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along a specific dimension.

    Args:
    ----
        input: Tensor to compute log-softmax on.
        dim: Dimension along which to compute log-softmax.

    Returns:
    -------
        Tensor with log-softmax applied.

    """
    max_values = max(input, dim)
    log_sum_exp = (input - max_values).exp().sum(dim).log()
    return input - log_sum_exp

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling to the input tensor.

    Args:
    ----
        input: Tensor with shape batch x channel x height x width.
        kernel: Tuple specifying the pooling kernel size (height, width).

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width after max pooling.
    
    """
    tiled_tensor, new_height, new_width = tile(input, kernel)
    max_tensor = max(tiled_tensor, dim=4)
    return max_tensor.view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input: Tensor to apply dropout on.
        p: Probability of dropping each element.
        training: Whether the model is in training mode (dropout applied).

    Returns:
    -------
        Tensor with elements dropped out.
    
    """
    if not training:
        return input
    mask = rand(input.shape) > p
    return input * mask
