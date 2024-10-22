import logging
import re
import unittest
from typing import Callable, Iterable, Union

import torch_npu

Ascend910B = "Ascend910B"
Ascend310P = "Ascend310P"
Ascend910 = "Ascend910"
Ascend310B = "Ascend310B"
Ascend910_93 = "Ascend910_93"

ALL_SOC_SET = set(
    Ascend910B, Ascend310P, Ascend310B, Ascend910, Ascend910_93
)


def get_soc_name() -> str:
    """
    获取芯片名称

    :return: 芯片名称
    """
    available_soc_list = (Ascend910B, Ascend310P)
    # 910注意顺序
    device_name = torch_npu.npu.get_device_name()
    for soc_name in available_soc_list:
        if re.search(soc_name, device_name, re.I):
            return soc_name
    logging.error("device_name %s is not supported", device_name)
    return None


def on_soc(soc_name: Union[str, Iterable[str]]) -> Callable:
    """
    限制运行在某芯片上的装饰器

    :param soc_name: 芯片名称
    :return: 芯片限制装饰器
    """
    if isinstance(soc_name, str):
        return unittest.skipIf(
            soc_name != get_soc_name(), f"This case only runs on {soc_name}"
        )
    return unittest.skipIf(
        get_soc_name(
        ) not in soc_name, f"This case only runs on {', '.join(soc_name)}"
    )
    
def skip_soc(soc_name: Union[str, Iterable[str]]) ->callable:
    if isinstance(soc_name, str):
        soc_name = (soc_name)
    return on_soc(ALL_SOC_SET - set(soc_name))


only_910b = on_soc(Ascend910B)
only_310p = on_soc(Ascend310P)
only_310b = on_soc(Ascend310B)
only_910 = on_soc(Ascend910)

skip_310p = skip_soc(Ascend310P)
skip_310b = skip_soc(Ascend310B)
skip_910 = skip_soc(Ascend910)
