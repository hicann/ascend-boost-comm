# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import Enum
from typing import Callable, TypeVar

import torch

K = TypeVar('K')
V = TypeVar('V')


def enum_to_dict(enum: Enum, key: Callable[[Enum], K] = lambda member: member.name,
                 value: Callable[[Enum], V] = lambda member: member.value) -> dict[K, V]:
    return {key(member): value(member) for member in enum}


class OpType(Enum):
    NA = 0
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9


# all possible type names
TENSOR_DTYPES_DICT = {
    **dict.fromkeys((0, 'float', 'float32', 'fp32'), torch.float32),
    **dict.fromkeys((1, 'half', 'float16', 'fp16'), torch.float16),
    **dict.fromkeys((2, 'char', 'int8'), torch.int8),
    **dict.fromkeys((3, 'long', 'int32',), torch.int32),
    **dict.fromkeys((4, 'unsigned char', 'uint8'), torch.uint8),
    **dict.fromkeys((6, 'short', 'int16'), torch.int16),
    **dict.fromkeys((7, 'unsigned short', 'uint16'), torch.uint16),
    **dict.fromkeys((8, 'unsigned long', 'uint32'), torch.uint32),
    **dict.fromkeys((9, 'long long', 'int64',), torch.int64),
    **dict.fromkeys((10, 'unsigned long long', 'uint64'), torch.uint64),
    **dict.fromkeys((11, 'double', 'float64', 'fp64'), torch.float64),
    **dict.fromkeys((12, 'bool'), torch.bool),
    # **dict.fromkeys((13, 'string'), str),
    **dict.fromkeys((16, 'complex64'), torch.complex64),
    **dict.fromkeys((17, 'complex128'), torch.complex128),
    **dict.fromkeys((27, 'bfloat16', 'bf16'), torch.bfloat16)
}


class TensorFormat(Enum):
    UNDEFINED = -1
    NCHW = 0
    NHWC = 1
    ND = 2
    NC1HWC0 = 3
    FRACTAL_Z = 4
    NC1HWC0_C04 = 12
    HWCN = 16
    NDHWC = 27
    FRACTAL_NZ = 29
    NCDHW = 30
    NDC1HWC0 = 32
    FRACTAL_Z_3D = 33


class ErrorType(Enum):
    UNDEFINED = -1,
    NO_ERROR = 0,
    ERROR_INVALID_VALUE = 1,
    ERROR_OPERATION_NOT_EXIST = 2,
    ERROR_TACTIC_NOT_EXIST = 3,
    ERROR_KERNEL_NOT_EXIST = 4,
    ERROR_ATTR_NOT_EXIST = 5,
    ERROR_ATTR_INVALID_TYPE = 6,
    ERROR_LAUNCH_KERNEL_ERROR = 7,
    ERROR_SYNC_STREAM_ERROR = 8,
    ERROR_INFERSHAPE_ERROR = 9,
    ERROR_NOT_CONSISTANT = 10


ERROR_DICT: dict[str, ErrorType] = enum_to_dict(ErrorType, value=lambda member: member)

ERROR_INFO: dict[int, ErrorType] = enum_to_dict(
    ErrorType, key=lambda member: member.value, value=lambda member: member)

TENSOR_FORMAT_DICT = enum_to_dict(TensorFormat,
                                  key=lambda member: member.name.lower(), value=lambda member: member)
