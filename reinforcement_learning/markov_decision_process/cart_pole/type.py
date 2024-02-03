"""This module provides tensor and numpy validators for validation inputs in Pydantic models. It also defines common tensor and numpy types annotated with those validators.

It includes the following validators:
- `tensor_validator(dim: int, dtype: torch.dtype)` which validates whether given input is an instance of `torch.Tensor` and if it has expected dimension and dtype.
- `numpy_validator(shape: tuple[int, ...], dtype: np.dtype)` which validates whether given input is an instance of `numpy.ndarray` and verifies its shape and dtype.

These validators return callable functions that can be used as Pydantic validators.

This module also defines the following typed aliases:
- `Tensor1DimFloat32` for 1-D Tensor of float32
- `Tensor2DimFloat32` for 2-D Tensor of float32
- `Tensor1DimInt32` for 1-D Tensor of int32
- `Tensor1DimInt64` for 1-D Tensor of int6
- `State` for numpy array with a shape of (4,) and dtype of float32

These typed aliases are annotated with `PlainValidator` applied to the corresponding validator, specifying the requirements of dimensionality and dtype.

The validators raise `PydanticCustomError` upon validation failure, with specific error message reflecting the type and nature of the validation error.
"""
from typing import Annotated, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from pydantic import PlainValidator
from pydantic_core import PydanticCustomError
from pydantic_core.core_schema import NoInfoValidatorFunction
from torch import Tensor


def tensor_validator(dim: int, dtype: torch.dtype) -> NoInfoValidatorFunction:
    """Returns a validator which checks a given object is a `torch.Tensor` instance and has the expected dimension and dtype.

    Args:
        dim: An integer specifying the expected number of dimensions of the input Tensor.
        dtype: The expected data type of the input Tensor.

    Returns:
        A Callable that can be used as a Pydantic validator for Tensors, ensuring that the input has the specified number of dimensions and data type.

    Raises:
        PydanticCustomError: If the input is not a Tensor, has a different number of dimensions, or has a different data type.
    """

    def validator(v: Any) -> Tensor:  # noqa: ANN401
        if not isinstance(v, Tensor):
            raise PydanticCustomError(
                "type error",
                "Expected a Tensor instance, but got a {actual_instance_type}",
                {"actual_instance_type": type(v)},
            )
        if v.dim() != dim:
            raise PydanticCustomError(
                "dimension mismatch error",
                "Expected a {expected_dim}-dim Tensor, but got {actual_dim}-dim Tensor",
                {"expected_dim": dim, "actual_dim": v.dim()},
            )
        if v.dtype != dtype:
            raise PydanticCustomError(
                "dtype error",
                "Expected {expected_dtype}, but got {actual_dtype}",
                {"expected_dtype": dtype, "actual_dtype": v.dtype},
            )
        return v

    return validator


Tensor1DimFloat32: TypeAlias = Annotated[Tensor, PlainValidator(tensor_validator(dim=1, dtype=torch.float32))]
Tensor2DimFloat32: TypeAlias = Annotated[Tensor, PlainValidator(tensor_validator(dim=2, dtype=torch.float32))]
Tensor1DimInt32: TypeAlias = Annotated[Tensor, PlainValidator(tensor_validator(dim=1, dtype=torch.int32))]
Tensor1DimInt64: TypeAlias = Annotated[Tensor, PlainValidator(tensor_validator(dim=1, dtype=torch.int64))]


def numpy_validator(shape: tuple[int, ...], dtype: np.dtype[Any]) -> NoInfoValidatorFunction:
    """Returns a validator which checks a given object is a `np.ndarray` instance and has the expected shape and dtype.

    Args:
        shape: A tuple of integers representing the expected shape of the numpy array.
        dtype: The expected data type of the numpy array.

    Returns:
        A validator function that validates whether the input value is a numpy array with the expected shape and data type.

    Raises:
        PydanticCustomError: If the input value is not a numpy array, or if the shape or data type of the numpy array is not as expected.
    """

    def validator(v: Any) -> npt.NDArray[Any]:  # noqa: ANN401
        if not isinstance(v, np.ndarray):
            raise PydanticCustomError(
                "type error",
                "Expected a numpy.ndarray instance, but got a {actual_instance_type}",
                {"actual_instance_type": type(v)},
            )
        if v.shape != shape:
            raise PydanticCustomError(
                "dimension mismatch error",
                "Expected a {expected_dim}-shape array, but got {actual_shape}-shape array",
                {"expected_shape": shape, "actual_array": v.shape},
            )
        if v.dtype != dtype:
            raise PydanticCustomError(
                "dtype error",
                "Expected {expected_dtype}, but got {actual_dtype}",
                {"expected_dtype": dtype, "actual_dtype": v.dtype},
            )
        return v

    return validator


State: TypeAlias = Annotated[npt.NDArray[Any], PlainValidator(numpy_validator(shape=(4,), dtype=np.float32))]
