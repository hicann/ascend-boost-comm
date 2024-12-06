# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

import torch


def log_tensors(tensors: list[torch.Tensor], tensor_formats=None, value: bool = False):
    if tensor_formats is None:
        tensor_formats = []
    for idx, tensor in enumerate(tensors):
        logging.debug("Tensor[%s]:", idx)
        if tensor_formats:
            logging.debug("Shape: %s,Dtype: %s, Format: %s",
                          tensor.shape, tensor.dtype, tensor_formats[idx])
        else:
            logging.debug("Shape: %s,Dtype: %s", tensor.shape, tensor.dtype)
        if value:
            logging.debug("Value:")
            logging.debug(tensor)
