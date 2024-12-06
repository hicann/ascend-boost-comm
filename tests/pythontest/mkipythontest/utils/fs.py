# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pathlib
from typing import Iterable, Union


def find_csv(csv_path: Union[str, Iterable[str]]) -> list[str]:
    """Find all csv path in a file path.

    :param csv_path: where to find the csv
    :return: absolute path list for csv
    """
    if isinstance(csv_path, str):
        path = pathlib.Path(csv_path)
        return [str(path)] if path.is_file() else list(map(str, pathlib.Path(path).glob('**/*.csv')))
    return [file for path in csv_path for file in find_csv(path)]
