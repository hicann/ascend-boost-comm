# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch


def is_float_type(dtype: torch.dtype) -> bool:
    """Check if a dtype is float.

    :param dtype: dtype
    :return: if it is float
    """
    return dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def is_int_type(dtype: torch.dtype) -> bool:
    """Check if a dtype is integer.

    :param dtype: dtype
    :return: if it is integer
    """
    return dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

def get_higher_precision_dtype(dtype: torch.dtype) -> torch.dtype:
    """Get the higher precision dtype of a dtype.

    :param dtype: dtype
    :return: higher precision dtype
    """
    if is_float_type(dtype):
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        if dtype == torch.float32:
            return torch.float64
    return dtype
