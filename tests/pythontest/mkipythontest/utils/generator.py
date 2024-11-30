# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import logging
from functools import lru_cache
from typing import Callable, Optional

import numpy
import torch
from mkipythontest.utils.misc import update_copy


def get_param_from_generator_str(generator_str: str) -> dict:
    """Get parameters from a generator string

    :param generator_str: generator string
    :return: parameters dict
    """
    tree = ast.parse(generator_str)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        result = {}
        for kw in node.keywords:
            value_raw_str = ast.unparse(kw.value)
            value_float = float(value_raw_str)
            if int(value_float) == value_float:
                result[kw.arg] = int(value_raw_str)
            else:
                result[kw.arg] = value_float
        return result
    return {}


@lru_cache
def get_func_and_param_from_generator_str(generator_str: str) -> tuple[Callable, dict]:
    """Get function and param from a generator string

    :param generator_str: generator string
    :return: generator function and param
    """
    func = numpy.random.uniform
    func_param = {
        'low': -5,
        'high': 5,
    }
    func_pkgs = (torch, numpy.random, numpy)
    func_param = get_param_from_generator_str(generator_str)
    func_name = generator_str.split('(')[0]
    for func_pkg in func_pkgs:
        func = getattr(func_pkg, func_name, None)
        if func is not None:
            break
    return func, func_param


@lru_cache
def get_generator(generator_str: str) -> Callable:
    func, param = get_func_and_param_from_generator_str(generator_str)
    shape_key = ''
    # numpy is a C library, so it is not possible to get its param name with inspect
    # we can only use exception
    for sk in ('shape', 'size'):
        try:
            shape_param = update_copy(param, {sk: (1,)})
            func(**shape_param)
            shape_key = sk
            break
        except TypeError as _:
            continue

    def generator_wrapper(shape: Optional[tuple[int]] = None) -> torch.Tensor:
        if not shape and shape_key:
            logging.error(f"func {func} accept shape param but not provided")
        if shape and not shape_key:
            logging.info(f"func {func} does not accept shape param")
        param = update_copy(param, {shape_key: shape})
        data = func(**param)
        if isinstance(data, numpy.ndarray):
            return torch.from_numpy(data)
        else:
            return data

    return generator_wrapper


def gen_tensor(dtype: torch.dtype, shape: tuple[int], generator_str: str) -> torch.Tensor:
    """Generate one tensor with generator string

    :param dtype: dtype
    :param shape: shape tuple
    :param generator_str: generator string
    :return: generated tensor
    """
    return get_generator(generator_str)(shape).to(dtype)


def gen_tensors(dtypes: list[torch.dtype],
                shapes: list[tuple[int]],
                generator_str_list: list[str]) -> list[torch.Tensor]:
    """Generate tensors with generator string list

    :param dtypes: dtype list
    :param shapes: shape tuple list
    :param generator_str_list: generator string list
    :return: generated tensor list
    """
    tensors = []
    for i in range(len(dtypes)):
        tensors.append(gen_tensor(dtypes[i], shapes[i], generator_str_list[i]))
    return tensors


def gen_empty_tensors(dtypes: list[torch.dtype], shapes: list[tuple[int]]) -> list[torch.Tensor]:
    """Generate tensors with zeros()

    :param dtypes: dtype list
    :param shapes: shape tuple list
    :return: generated tensor list with all zeros
    """
    return [torch.zeros(shape, dtype=dtype) for dtype, shape in zip(dtypes, shapes)]
