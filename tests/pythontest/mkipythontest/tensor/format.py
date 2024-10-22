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
import logging
from typing import Type

import torch


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


def from_nd(tensor: torch.Tensor, to_format: Type[TensorFormat], **kwargs) -> torch.Tensor:
    """
    tensor格式转换

    :param tensor:
    :param from_format:
    :param to_format:
    :return:
    """
    shape = tensor.shape
    if to_format == TensorFormat.ND:
        return tensor
    if to_format == TensorFormat.FRACTAL_NZ:
        return tensor
        
    logging.info(f"Format {to_format.name} is not supported now.")
    return tensor
