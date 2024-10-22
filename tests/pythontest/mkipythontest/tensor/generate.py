# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import ast
from functools import lru_cache
from inspect import signature

import numpy
import torch


def get_generator_param(generator_str: str) -> dict:
    """
    获取生成器参数

    :param generator_str: 生成器字符串
    :return: 生成器参数
    """
    tree = ast.parse(generator_str)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        return {kw.arg: float(ast.unparse(kw.value))
                for kw in node.keywords}
    return {}


@lru_cache
def func_call_str_to_func_param(generator_str: str) -> tuple[callable, dict]:
    """
    获取生成器函数

    :param generator_str: 生成器字符串
    :return: 生成器函数
    """
    func = numpy.random.uniform
    func_param = {
        'low': -5,
        'high': 5,
    }
    func_pkgs = (torch, numpy.random, numpy)
    func_param = get_generator_param(generator_str)
    func_name = generator_str.split('(')[0]
    for func_pkg in func_pkgs:
        func = getattr(func_pkg, func_name, None)
        if func is not None:
            break
    return func, func_param


def gen_tensor(dtype: torch.dtype, shape: tuple[int], generator_str: str) -> torch.Tensor:
    func, param = func_call_str_to_func_param(generator_str)
    func_signature = signature(func)
    if 'shape' in func_signature.parameters:
        param.update('shape', shape)
    elif 'size' in func_signature.parameters:
        param.update('size', shape)
    raw_data = func(**param)
    if isinstance(raw_data, numpy.ndarray):
        raw_data = torch.from_numpy(raw_data)
    return raw_data.to(dtype)


def gen_tensors(dtypes: list[torch.dtype], shapes: list[tuple[int]], generator_str_list: list[str]) -> list[torch.Tensor]:
    """
    用生成器生成张量

    :param dtypes: 张量数据类型
    :param shapes: 张量形状
    :param generators: 生成器函数列表
    :return: 生成张量列表
    """
    tensors = []
    for i in len(dtypes):
        tensors.append(gen_tensor(dtypes[i], shapes[i], generator_str_list[i]))
    return tensors


def gen_empty_tensors(dtypes: list[torch.dtype], shapes: list[tuple[int]]) -> list[torch.Tensor]:
    """
    生成空张量

    :param dtypes: 张量数据类型
    :param shapes: 张量形状
    :return: 全0张量列表
    """
    generator_str_list = ["zeros()"] * len(dtypes)
    return gen_tensors(dtypes, shapes, generator_str_list)
