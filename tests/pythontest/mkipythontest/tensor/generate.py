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
from functools import partial

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


def get_generator(generator_str: str) -> callable:
    """
    获取生成器函数

    :param generator_str: 生成器字符串
    :return: 生成器函数
    """
    func = numpy.random.uniform
    func_kwargs = {
        'low': -5,
        'high': 5,
    }
    func_pkgs = (torch, numpy.random, numpy)
    func_kwargs = get_generator_param(generator_str)
    func_name = generator_str.split('(')[0]
    for func_pkg in func_pkgs:
        func = getattr(func_pkg, func_name, None)
        if func is not None:
            break
    return partial(func, **func_kwargs)


def gen_empty_tensors(dtypes: list[torch.dtype], shapes: list[tuple[int]])->list[torch.Tensor]:
    """
    生成空张量

    :param dtypes: 张量数据类型
    :param shapes: 张量形状
    :return: 全0张量列表
    """
    return gen_tensors(dtypes, shapes, [get_generator("zeros()") for _ in range(len(shapes))])


def gen_tensors(dtypes: list[torch.dtype], shapes: list[tuple[int]], generators: list[callable])->list[torch.Tensor]:
    """
    用生成器生成张量

    :param dtypes: 张量数据类型
    :param shapes: 张量形状
    :param generators: 生成器函数列表
    :return: 生成张量列表
    """
    tensors = []
    for i in range(len(dtypes)):
        dtype = dtypes[i]
        shape = shapes[i]
        generator = generators[i]
        in_tensor = generator(size=shape)
        if isinstance(in_tensor, numpy.ndarray):
            in_tensor = torch.from_numpy(in_tensor)
        tensors.append(in_tensor.to(dtype))
    return tensors
