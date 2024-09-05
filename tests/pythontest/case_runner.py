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
import numpy
from op_test import OpTest
from constant import OP_DTYPES
import torch
import pandas as pd


def __test_runner(self: OpTest, case_name: str) -> None:
    # get case from self
    case_line = self.test_cases[case_name]["case"]
    # set soc version
    soc_version = case_line['SocVersion']
    if soc_version == 'Ascend910B':
        self.set_support_910b_only()
    if soc_version == 'Ascend310P':
        self.set_support_310p_only()
    # set param
    op_name = case_line['OpName']
    op_param = json.loads(case_line['OpParam'])
    self.set_param(op_name, op_param)
    # gen tensor
    random_seed = random.randint(0, 1000000)
    numpy.random.seed(random_seed)

    in_num = int(case_line['InNum'])
    in_dtype = tuple(
        map(lambda t: OP_DTYPES[t], case_line['InDType'].split(';')))
    in_format = tuple(case_line['InFormat'].split(';'))
    in_shape = tuple(
        map(lambda s: tuple(map(int, s.split(','))), str(case_line['InShape']).split(';')))
    in_tensors = []
    data_gen_types = tuple(case_line['DataGeneration'].split(';'))
    for i in range(in_num):
        data_gen_type = data_gen_types[i]
        in_tensor = None
        if data_gen_type == 'custom':
            in_tensor = self.custom(in_dtype, in_format, in_shape)
        else:
            shape = in_shape
            dtype = in_dtype
            in_tensor = torch.from_numpy(eval(data_gen_type)).to(dtype)
        in_tensors.append(in_tensor)
    out_num = int(case_line['OutNum'])
    out_dtype = tuple(
        map(lambda t: OP_DTYPES[t], case_line['OutDType'].split(';')))
    out_format = tuple(case_line['OutFormat'].split(';'))
    out_shape = tuple(
        map(lambda s: tuple(map(int, s.split(','))), str(case_line['OutShape']).split(';')))
    out_tensors = []
    for i in range(out_num):
        out_tensor = torch.zeros(
            out_shape[i]).to(out_dtype[i])
        out_tensors.append(out_tensor)
    # execute
    test_type = case_line['TestType']
    case_file = self.test_cases[case_name]['file_path']
    result_data = pd.read_csv(case_file, sep='|')
    result_data = pd.concat([result_data, pd.DataFrame(
        columns=["mare", "mere", "rmse", "err", "eb", "pass", "RandomSeed"])], sort=False)
    if test_type == 'Function':
        # 仅关心是否通过，使用ut的assert
        self.execute(in_tensors, out_tensors)
