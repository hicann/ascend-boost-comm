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

from mkipythontest.case import Case
from mkipythontest.constant import TestType


def evaluate_expression(expression: str, mappings: dict[str, int]):
    """
    计算数学表达式

    :param expression: 字符串形式的数学表达式
    :param mappings: 一个字典，包含占位符到具体值的映射
    :return: 表达式的计算结果
    """
    return eval(expression, {"__builtins__": None}, mappings)


def discover_placeholders(shapes: list[tuple],
                          op_param: dict,
                          placeholder_pattern: str = r'^[a-zA-Z_][a-zA-Z0-9_]*$') -> set[str]:
    """
    获取所有占位符

    :param placeholder_pattern: 占位符格式
    :return: 占位符集合
    """
    placeholders = set()
    for shape in shapes:
        for dim in shape:
            if isinstance(dim, int):
                continue
            for matched_placeholder in re.finditer(placeholder_pattern, str(dim)):
                placeholders.add(matched_placeholder.string)
    for _, value in op_param.items():
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


def dict_to_name_postfix(mappings: dict[str, int]) -> str:
    result = []
    for key in mappings:
        result.append(str(key))
        result.append(str(mappings[key]))
    return "_".join(result)


def make_generalized_shapes_and_op_param(base_case: Case,
                                         in_shapes: list[tuple],
                                         out_shapes: list[tuple],
                                         op_param: dict = {},
                                         generalized_dims=None) -> list[Case]:
    """
    生成全排列泛化测试用例

    :param generalized_dims: 泛化维度列表
    :return:
    """
    if generalized_dims is None:
        generalized_dims = [1, 4, 15, 16, 17,
                            32, 64, 65, 128, 256, 257, 131073]
    shapes = in_shapes+out_shapes
    placeholders = discover_placeholders(shapes, op_param)
    mapping_combinations = list(itertools.product(
        generalized_dims, repeat=len(placeholders)))
    result = []
    for mapping in mapping_combinations:
        mappings = dict(zip(placeholders, mapping))
        new_shapes = []
        for shape in shapes:
            dynamic_shape = []
            for s in shape:
                dynamic_shape.append(evaluate_expression(str(s), mappings))
            new_shapes.append(tuple(dynamic_shape))
        new_param = {}
        for k, v in op_param.items():
            new_param[k] = evaluate_expression(str(v), mappings)
        new_case = deepcopy(base_case)
        new_case.set_name(
            f"{new_case.case_name}_generalized_{dict_to_name_postfix(mappings)}")
        new_case.op_param = new_param
        new_case.in_shapes = new_shapes[0:new_case.in_num]
        new_case.out_shapes = new_shapes[new_case.in_num:]
        new_case.test_type = TestType.FUNCTION
        result.append(new_case)
    return result
