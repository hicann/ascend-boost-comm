# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from enum import Enum

import numpy
import torch


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


class TestType(Enum):
    FUNCTION = 0
    GENERALIZATION = 1
    PERFORMANCE = 2


# all possible type names
TENSOR_DTYPES = {
    **dict.fromkeys((0, 'float', 'float32', 'fp32'), torch.float32),
    **dict.fromkeys((1, 'half', 'float16', 'fp16'), torch.float16),
    **dict.fromkeys((2, 'char', 'int8'), torch.int8),
    **dict.fromkeys((3, 'long', 'int32',), torch.int32),
    **dict.fromkeys((4, 'unsigned char', 'uint8'), torch.uint8),
    **dict.fromkeys((6, 'short', 'int16'), torch.int16),
    **dict.fromkeys((7, 'unsigned short', 'uint16'), numpy.uint16),
    **dict.fromkeys((8, 'unsigned long', 'uint32'), numpy.uint32),
    **dict.fromkeys((9, 'long long', 'int64',), torch.int64),
    **dict.fromkeys((10, 'unsigned long long', 'uint64'), numpy.uint64),
    **dict.fromkeys((11, 'double', 'float64', 'fp64'), torch.float64),
    **dict.fromkeys((12, 'bool'), torch.bool),
    **dict.fromkeys((13, 'string'), numpy.bytes_),
    **dict.fromkeys((16, 'complex64'), torch.complex64),
    **dict.fromkeys((17, 'complex128'), torch.complex128),
    **dict.fromkeys((27, 'bfloat16', 'bf16'), torch.bfloat16)
}


class ErrorType(Enum):
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
