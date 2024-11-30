# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os


def set_envs(env=None):
    """Set environment variables.

    :param env: environment variables, defaults to {}
    """
    if env is None:
        env = {}
    for key, value in env.items():
        os.environ[key] = value


def unset_envs(env=None):
    """Unet environment variables.

    :param env: environment variables, defaults to {}
    """
    if env is None:
        env = {}
    for key, _ in env.items():
        os.environ[key] = ""


def get_npu_device() -> str:
    """Select running NPU.

    :return: npu set string
    """
    npu_device = os.environ.get("MKI_NPU_DEVICE")
    if npu_device is None:
        npu_device = "npu:0"
    else:
        npu_device = f"npu:{npu_device}"
    return npu_device
