# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch

# NZ byte num
BLOCK_SIZE = 32


def pad_to_divisible(num: int, divisor: int, floor: bool = False) -> int:
    """pad a number to be divisible by a divisor

    :param num: input num
    :param divisor: the divisor
    :param floor: whether to floor or ceil, defaults to False
    :return: padded number
    """
    if num % divisor == 0:
        return num
    else:
        return num + (divisor - num % divisor) if floor else num + divisor - num % divisor


def convert_nd_to_fractal_nz(tensor: torch.Tensor, merge_axis_h: bool = True) -> torch.Tensor:
    """Convert ND tensor to Fractal NZ tensor

    :param tensor: input tensor
    :param merge_axis_h: whether to merge H axis, defaults to False
    :return: nz tensor
    """
    # (..., N, W1,H1, H0, W0)->transpose->(..., N, H1, H0, W1, W0)->reshape->(..., N, H1*H0, W1*W0)->unpad->(..., N，H, W )
    def get_transpose_tuple(origin_ndim: int) -> tuple[int, ...]:
        offset = origin_ndim-2
        return tuple([x for x in range(offset)]+[x+offset for x in [2, 0, 1, 3]])

    shape = tensor.shape
    h = shape[-2]
    w = shape[-1]
    h0, w0 = 16, int(BLOCK_SIZE/tensor.dtype.itemsize)
    h_pad_num = pad_to_divisible(h, h0)-h
    w_pad_num = pad_to_divisible(w, w0)-w
    # pad
    tensor = torch.nn.functional.pad(
        tensor, pad=[0, w_pad_num, 0, h_pad_num])
    h_padded = tensor.shape[-2]
    w_padded = tensor.shape[-1]
    # reshape
    h1 = int(h_padded/h0)
    w1 = int(w_padded/w0)
    tensor = tensor.reshape(shape[:-2]+(h1, h0, w1, w0))
    # transpose
    tensor = tensor.permute(
        get_transpose_tuple(origin_ndim=tensor.ndim))
    if merge_axis_h:
        return tensor.reshape(tensor.shape[:-4]+(w1, h_padded, w0))
    return tensor


def convert_fractal_nz_to_nd(tensor: torch.Tensor, 
                             merge_axis_h: bool = True, 
                             unpad_h_num: int = 0, 
                             unpad_w_num: int = 0) -> torch.Tensor:
    """Convert Fractal NZ tensor to ND tensor

    :param tensor: input tensor
    :param merge_axis_h: whether to merge H axis, defaults to False
    :return: nd tensor
    """
    # (..., N，H, W )->pad->(..., N, H1*H0, W1*W0)->reshape->(..., N, H1, H0, W1, W0)->transpose->(..., N, W1,H1, H0, W0)
    def get_transpose_tuple(offset: int) -> tuple[int, ...]:
        return tuple([x for x in range(offset)]+[x+offset for x in [1, 2, 0, 3]])
    if merge_axis_h:
        w0, w1 = tensor.shape[-1], tensor.shape[-3]
        h0, h1 = 16, int(tensor.shape[-2]/16)
        tensor = tensor.reshape(tensor.shape[:-3]+(w1, h1, h0, w0))
    else:
        w1, h1, h0, w0 = tensor.shape[-4:]
    tensor = tensor.permute(get_transpose_tuple(tensor.ndim-4))
    tensor = tensor.reshape(tensor.shape[:-4]+(int(h1*h0), int(w1*w0)))
    if unpad_h_num != 0:
        tensor = tensor.narrow(-2, 0, tensor.shape[-2]-unpad_h_num)
    if unpad_w_num != 0:
        tensor = tensor.narrow(-1, 0, tensor.shape[-1]-unpad_w_num)
    return tensor
