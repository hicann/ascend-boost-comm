# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from functools import lru_cache

import torch
from mkipythontest.constant import ErrorType, OpType, TestType


class Case:
    """
    测试用例
    """

    def __init__(self, specific_case_name: str = "", in_num: int = 1, out_num: int = 1) -> None:
        self.__case_name: str = specific_case_name
        self.op_name: str = ""
        self.op_type: OpType = OpType.NA
        self.op_param: dict = {}

        self.in_dtypes: list[torch.dtype] = [
            torch.float16 for _ in range(in_num)]
        self.in_shapes: list[tuple[int]] = [(1,) for _ in range(in_num)]
        self.in_formats: list[str] = ["nd" for _ in range(in_num)]

        self.out_dtypes: list[torch.dtype] = [
            torch.float16 for _ in range(out_num)]
        self.out_shapes: list[tuple[int]] = [(1,) for _ in range(out_num)]
        self.out_formats: list[str] = ["nd" for _ in range(out_num)]

        self.data_generate: str = ""

        self.soc_version: list[str] = []
        self.test_type: TestType = TestType.FUNCTION
        self.expected_error: ErrorType = ErrorType.NO_ERROR
        self.env: dict = {}
        self.dump: bool = False
        self.random_seed = 0

    @property
    def case_name(self):
        if self.__case_name:
            return self.__case_name
        else:
            return f"{self.op_name}_" + "_".join(tuple(map(str, (self.in_shapes[0]))))

    def set_name(self, name: str):
        self.__case_name = name

    @property
    def in_num(self):
        return len(self.in_dtypes)

    @property
    def out_num(self):
        return len(self.out_dtypes)

    @property
    def all_shapes(self):
        return self.in_shapes + self.out_shapes

    @property
    @lru_cache
    def in_tensors(self):
        return [torch.zeros(self.in_shapes[i]).to(self.in_dtypes[i]) for i in range(self.in_num)]

    @property
    @lru_cache
    def out_tensors(self):
        return [torch.zeros(self.out_shapes[i]).to(self.out_dtypes[i]) for i in range(self.out_num)]
