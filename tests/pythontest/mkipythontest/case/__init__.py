# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import itertools
import logging
import re
from copy import deepcopy

import torch
from mkipythontest.constant import OpType, TestType, ErrorType


class Case:
    """
    测试用例
    """

    def __init__(self, specific_case_name: str = "") -> None:
        self.__case_name: str = specific_case_name
        self.op_name: str = ""
        self.op_type: OpType = OpType.NA
        self.op_param: dict = {}

        self.in_dtypes: list[torch.dtype] = []
        self.in_shapes: list[tuple[int]] = []
        self.in_formats: list[str] = []

        self.out_dtypes: list[torch.dtype] = []
        self.out_shapes: list[tuple[int]] = []
        self.out_formats: list[str] = []

        self.data_generate: str = ""
        self.soc_version: list[str] = []
        self.test_type: TestType = TestType.FUNCTION
        self.expected_error: ErrorType = ErrorType.NO_ERROR
        self.env: dict = {}

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


class GeneralizedCase(Case):
    """具有占位符的泛化测试用例"""

    def __evaluate_expression(self, expression: str, mappings: dict[str, int]):
        """
        计算数学表达式

        :param expression: 字符串形式的数学表达式
        :param mappings: 一个字典，包含占位符到具体值的映射
        :return: 表达式的计算结果
        """
        expression = expression.replace(" ", "")
        expression_placeholder_and_number = re.split(
            r'[+\-*/]|/\|\\*\\*', expression)
        expression_operator = re.findall(r'[+\-*/]|/\|\\*\\*', expression)
        for i, placeholder_and_number in enumerate(expression_placeholder_and_number):
            if not placeholder_and_number.isdigit():
                expression_placeholder_and_number[i] = mappings[placeholder_and_number]
        expression = ''.join(
            map(str, sum(zip(expression_placeholder_and_number, expression_operator + ["dummy"]), ())[0:-1]))
        return int(eval(expression))

    def __discover_placeholders(self, placeholder_pattern: str = r'^[a-zA-Z_][a-zA-Z0-9_]*$') -> set[str]:
        """
        获取所有占位符

        :param placeholder_pattern: 占位符格式
        :return: 占位符集合
        """
        placeholders = set()
        all_shapes = self.in_shapes + self.out_shapes
        for shape in all_shapes:
            for dim in shape:
                if isinstance(dim, int):
                    continue
                for matched_placeholder in re.finditer(placeholder_pattern, str(dim)):
                    placeholders.add(matched_placeholder.string)
        for _, value in self.op_param.items():
            if isinstance(value, int):
                continue
            objs = value
            if isinstance(value, str):
                objs = [value]
            for obj in objs:
                if match_str := re.match(placeholder_pattern, obj):
                    placeholders.add(match_str.string)
        logging.info("found placeholders: %s", placeholders)
        return placeholders

    def __dict_to_name_postfix(self, mappings: dict[str, int]) -> str:
        result = []
        for key in mappings:
            result.append(str(key))
            result.append(str(mappings[key]))
        return "_".join(result)

    def to_case_list(self, generalized_dims=None) -> list[Case]:
        """
        生成全排列泛化测试用例

        :param generalized_dims: 泛化维度列表
        :return:
        """
        if generalized_dims is None:
            generalized_dims = [1,4,15,16,17,32,64,65,128,256,257,131073]
        placeholders = self.__discover_placeholders()
        # 生成所有可能的映射表组合
        mapping_combinations = list(itertools.product(
            generalized_dims, repeat=len(placeholders)))
        # 替换占位符并生成所有可能的组合
        result = []
        for mapping in mapping_combinations:
            mappings = dict(zip(placeholders, mapping))
            new_shapes = []
            for shape in self.all_shapes:
                dynamic_shape = []
                for s in shape:
                    dynamic_shape.append(self.__evaluate_expression(str(s), mappings))
                new_shapes.append(tuple(dynamic_shape))
            new_param = {}
            for k, v in self.op_param.items():
                new_param[k] = self.__evaluate_expression(str(v), mappings)
            new_case = deepcopy(self)
            new_case.set_name(
                f"{new_case.case_name}_generalized_{self.__dict_to_name_postfix(mappings)}")
            new_case.op_param = new_param
            new_case.in_shapes = new_shapes[0:self.in_num]
            new_case.out_shapes = new_shapes[self.in_num:]
            new_case.test_type = TestType.FUNCTION
            result.append(new_case)
        return result


class BinCase(Case):
    """
    测试用例，从二进制文件中读取输入输出数据
    """
    pass
