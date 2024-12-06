# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import itertools
import logging
import re
from copy import deepcopy
from typing import Iterable, Optional, Union

from mkipythontest.case import Case

PLACEHOLDER_PATTERN: str = r'^[a-zA-Z_][a-zA-Z0-9_]*'


def evaluate_expression(expression: str, mapping: dict[str, int]):
    """Calculate an expression with given symbol mapping.

    :param expression: expression string
    :param mapping: mapping dict
    :return: the result of the expression
    """
    return eval(expression, {"__builtins__": None}, mapping)


def discover_placeholders_from_shapes(shapes: list[tuple], placeholder_pattern: str = PLACEHOLDER_PATTERN) -> set[str]:
    """Find placeholders from a list of shapes.

    :param shapes: a list contains shapes
    :param placeholder_pattern: regex expression of placeholder, defaults to PLACEHOLDER_PATTERN
    :return: found placeholders
    """
    placeholders = set()
    for shape in shapes:
        for dim in shape:
            for matched_placeholder in re.findall(placeholder_pattern, str(dim)):
                placeholders.add(matched_placeholder)
    return placeholders


def discover_placeholders_from_op_param(op_param: dict, placeholder_pattern: str = PLACEHOLDER_PATTERN) -> set[str]:
    """Find placeholders from a dict of operation param.

    :param op_param: a dict of operation param
    :param placeholder_pattern: regex expression of placeholder, defaults to PLACEHOLDER_PATTERN
    :return: found placeholders
    """
    placeholders = set()
    for _, values in op_param.items():
        if not isinstance(values, Iterable):
            values = [values]
        for value in values:
            for matched_placeholder in re.findall(placeholder_pattern, str(value)):
                placeholders.add(matched_placeholder)
    return placeholders


def discover_placeholders(shapes: list[tuple],
                          op_param: dict, placeholder_pattern: str = PLACEHOLDER_PATTERN
                          ) -> set[str]:
    """Find placeholders in shapes and operation param.

    :param shapes: a list contains shapes
    :param op_param: a dict of operation param
    :param placeholder_pattern: regex expression of placeholder, defaults to PLACEHOLDER_PATTERN
    :return: found placeholders
    """
    result_shapes = discover_placeholders_from_shapes(
        shapes, placeholder_pattern)
    result_op_params = discover_placeholders_from_op_param(
        op_param, placeholder_pattern)
    placeholders = result_shapes.union(result_op_params)
    logging.info("found placeholders: %s", placeholders)
    return placeholders


def dict_to_name_postfix(mapping: dict[str, int]) -> str:
    """Convert mapping to a string for case name

    :param mapping: mapping dict
    :return: a string with mapping info for case name
    """
    result = []
    for key in mapping:
        result.append(str(key))
        result.append(str(mapping[key]))
    return "_".join(result)


def generate_mappings(key_list: list[str],
                      generalization_value_dict: dict[str, list[int]],
                      default_dims=None) -> list[dict[str, int]]:
    if default_dims is None:
        default_dims = [1, 4, 15, 16, 17, 32, 64, 65, 128, 256, 257, 131073]
    for key in key_list:
        if key not in generalization_value_dict:
            generalization_value_dict[key] = default_dims
    return [dict(zip(key_list, p)) for p in itertools.product(*generalization_value_dict.values())]


def make_generalized_case_list(base_case: Case,
                               in_shapes: list[tuple],
                               out_shapes: list[tuple],
                               op_param=None,
                               generalized_dims=Optional[Union[list[int], dict[str, list[int]]]]) -> list[Case]:
    """Generate generalized case list from base case, shapes and operation param with generalization placeholders

    :param base_case: base case containing other params
    :param in_shapes: in shape list containing generalization placeholders
    :param out_shapes: out shape list containing generalization placeholders
    :param op_param: operation param  containing generalization placeholders
    :param generalized_dims: the range of generalization dims
    :return: a list of generated cases
    """

    if op_param is None:
        op_param = {}
    shapes = in_shapes + out_shapes
    placeholders = discover_placeholders(shapes, op_param)
    if not generalized_dims:
        generalized_dims = {}

    mapping_combinations = generate_mappings(
        list(placeholders), generalized_dims)
    result = []
    for mapping in mapping_combinations:
        # 泛化shape
        new_shapes = []
        for shape in shapes:
            dynamic_shape = []
            for s in shape:
                dynamic_shape.append(evaluate_expression(str(s), mapping))
            new_shapes.append(tuple(dynamic_shape))
        # 泛化param
        new_param = {}
        for k, v in op_param.items():
            if isinstance(v, Iterable):
                new_param[k] = list[map(
                    lambda ss: evaluate_expression(str(ss), mapping), v)]
            else:
                new_param[k] = evaluate_expression(str(v), mapping)
        new_case = deepcopy(base_case)
        new_case.set_name(
            f"{new_case.case_name}_generalized_{dict_to_name_postfix(mapping)}")
        new_case.op_param = new_param
        new_case.in_shapes = new_shapes[0:new_case.in_num]
        new_case.out_shapes = new_shapes[new_case.in_num:]
        result.append(new_case)
    return result
