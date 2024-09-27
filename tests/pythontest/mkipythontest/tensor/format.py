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
from typing import Type

import torch


class TensorFormat:
    id_ = -1

    @staticmethod
    def to_nd(tensor: torch.Tensor) -> torch.Tensor:
        """
        将tensor转为ND格式
        :param tensor:
        :return:
        """
        pass

    @staticmethod
    def from_nd(tensor: torch.Tensor) -> torch.Tensor:
        """
        将ND格式Tensor转为此格式
        :param tensor:
        :return:
        """
        pass


class ND(TensorFormat):
    id_ = 2

    def to_nd(self):
        return self

    def from_nd(self):
        return self


class FRACTAL_NZ(TensorFormat):
    id_ = 29
    pass


class TensorFormats(Enum):
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


TENSOR_FORMATS = {
    **dict.fromkeys(('nd', ND.id_), ND),
    **dict.fromkeys(('fractal_nz', FRACTAL_NZ.id_), FRACTAL_NZ),
}


def convert_format(tensor: torch.Tensor, from_format: Type[TensorFormat], to_format: Type[TensorFormat]):
    """
    tensor格式转换

    :param tensor:
    :param from_format:
    :param to_format:
    :return:
    """
    nd_tensor = from_format.to_nd(tensor)
    return to_format.from_nd(nd_tensor)
