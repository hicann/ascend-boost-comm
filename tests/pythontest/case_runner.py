# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
import random
from op_test import OpTest
from case import Case
import numpy
import torch
import pandas as pd

def __test_runner(self: OpTest, case_name: str) -> None:
    # get case from self
    case: Case = self.test_cases[case_name]
    # set soc version
    soc_version = case.soc_version
    if soc_version == 'Ascend910B':
        self.set_support_910b_only()
    if soc_version == 'Ascend310P':
        self.set_support_310p_only()
    # set param
    self.set_param(case.op_name, case.op_param)
    # gen tensor
    random_seed = random.randint(0, 1000000)
    numpy.random.seed(random_seed)

    in_tensors = []

    for i, in_tensor in enumerate(case.in_tensors):
        if in_tensor['generate'] == 'custom':
            in_tensor = self.custom(
                i, in_tensor['dtype'], in_tensor['format'], in_tensor['shape'])
        else:
            shape = in_tensor['shape']
            in_tensor = torch.from_numpy(
                eval(in_tensor['generate'])).to(in_tensor['dtype'])
        in_tensors.append(in_tensor)

    out_tensors = []
    for out_tensor in case.out_tensors:
        out_tensors.append(torch.zeros(
            out_tensors['shape']).to(out_tensors['dtype']))

    # execute
    test_type = case.test_type
    if test_type == 'Function':
        # 仅关心是否通过，使用ut的assert
        self.execute(in_tensors, out_tensors)
