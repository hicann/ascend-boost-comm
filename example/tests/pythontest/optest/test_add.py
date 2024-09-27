# 
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# AscendOpCommonLib is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
import torch
import unittest

from mkipythontest import OpTest
from mkipythontest.case import Case
from mkipythontest.case.injector import case_inject

@case_inject
class TestElewiseOperationAdd(OpTest):
    def calculate_times(self, in_tensors: list[torch.Tensor]) -> int:
        return in_tensors[0].numel()

    def golden_calc(self, in_tensors: list[torch.Tensor])->list[torch.Tensor]:
        return [torch.add(in_tensors[0], in_tensors[1])]

    def test_method1(self):
        self.set_param("ElewiseOperation", {"elewiseType":8})
        in_tensors = torch.rand(size=(2,3)),torch.rand(size=(2,3))
        out_tensors = [torch.zeros(size=(2,3))]
        self.execute(in_tensors, out_tensors)

    def test_method2(self):
        case = Case("method2")
        case.op_name = "ElewiseOperation"
        case.op_param = {"elewiseType":8}
        case.in_shapes = [(2,3),(2,3)]
        case.out_shapes = [(2,3)]
        case.in_dtypes = [torch.float,torch.float]
        case.out_dtypes = [torch.float]
        self.run_case(case)


if __name__ == '__main__':
    unittest.main()
