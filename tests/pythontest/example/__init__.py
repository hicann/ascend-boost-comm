# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import sys
sys.path.append("..")
import torch
from case.injector import CaseInject
import unittest
from op_test import OpTest


@CaseInject
class Example(OpTest):

    def golden_calc(self, in_tensors):
        return in_tensors
    
    def golden_compare(self, out_tensors, golden_out_tensors):
        return torch.allclose(out_tensors, golden_out_tensors)


if __name__ == "__main__":
    unittest.main()
