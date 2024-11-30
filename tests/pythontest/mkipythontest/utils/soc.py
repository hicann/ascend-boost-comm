# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import re
import unittest
from enum import Enum
from functools import lru_cache
from typing import Callable, Iterable, Optional

import torch_npu


class SocType(Enum):
    Ascend910B = 0
    Ascend310P = 1
    Ascend910 = 2


SOC_SUPPORT = {SocType.Ascend910B, SocType.Ascend310P, SocType.Ascend910}


@lru_cache
def get_soc_type(device_name: str = None) -> Optional[SocType]:
    """Get precise soc type.

    :return: soc type
    """
    # Hint: Actually, "Ascend910B" is Ascend910(A).
    # If there is other character behind "Ascend910B", it is Ascend910B.
    if device_name is None:
        device_name = torch_npu.npu.get_device_name()
    if re.search(r"Ascend910B\w+", device_name, re.I):
        return SocType.Ascend910B
    if re.search("Ascend910_93",device_name, re.I):
        return SocType.Ascend910B
    # future 910* soc add here
    if re.search("Ascend910", device_name, re.I):
        return SocType.Ascend910
    if re.search("Ascend310P", device_name, re.I):
        return SocType.Ascend310P
    logging.error("device_name %s is not supported", device_name)
    return None


def on_soc(soc_type: Iterable[SocType]) -> Callable:
    """Limit a case to run on specific soc(s).

    :param soc_type: soc type
    :return: a decorator function for case function
    """
    return unittest.skipIf(
        get_soc_type(
        ) not in soc_type, f"This case only runs on {', '.join(map(lambda st: st.name, soc_type))}"
    )


def skip_soc(soc_type: Iterable[SocType]) -> Callable:
    """Avoid case running on specific soc(s).

    :param soc_type: soc type
    :return: a decorator function for case function
    """
    return on_soc(SOC_SUPPORT - set(soc_type))


only_910b = on_soc((SocType.Ascend910B,))
only_310p = on_soc((SocType.Ascend310P,))
only_910 = on_soc((SocType.Ascend910,))

skip_310p = skip_soc((SocType.Ascend310P,))
skip_910 = skip_soc((SocType.Ascend910,))
