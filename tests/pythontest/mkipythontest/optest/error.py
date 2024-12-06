# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import functools
import logging

from mkipythontest.constant import ErrorType


class MkiOpTestError(RuntimeError):
    def __init__(self, error_type: ErrorType = ErrorType.UNDEFINED):
        self.error_type: ErrorType = error_type


def expect_error(error_type: ErrorType):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if isinstance(e, MkiOpTestError):
                    if e.error_type == error_type:
                        logging.info(
                            f"Raised expected exception: {e.error_type}")
                        return True
                    else:
                        raise AssertionError(
                            f"Raised unexpected exception: {e.error_type}")
                else:
                    raise AssertionError(
                        f"Expected MkiOpTestError but got {type(e).__name__}")
        return wrapper
    return decorator
