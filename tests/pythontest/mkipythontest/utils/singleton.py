# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from functools import wraps
from typing import Callable, Type


def Singleton(cls: object) -> Callable:
    """Make a class a Singleton class

    :param cls: class to be singleton
    :return: wrapped class
    """
    instances: dict[Type, object] = {}

    @wraps(cls)
    def get_instance(*args, **kwargs) -> object:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def RunOnce(func: Callable) -> Callable:
    """Make a function run only once

    :param func: function to be run only once
    :return: wrapped function
    """
    has_run = False
    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)
    return wrapper

