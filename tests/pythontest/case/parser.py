# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from case import Case
import pandas as pd
import json
from constant import OP_DTYPES
import ast
import numpy
from functools import partial
from abc import ABCMeta, abstractmethod
import torch


class BaseParser(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, csv_file_path: str) -> Case:
        pass


class DefaultCsvParser(BaseParser):
    def __get_generator(self, generator_str: str):
        func_pkgs = (torch,numpy.random,numpy)
        tree = ast.parse(generator_str)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = None
            for func_pkg in func_pkgs:
                func = getattr(func_pkg, node.func.id, None)
                if func is not None:
                    break
            kwargs = {kw.arg: float(ast.unparse(kw.value))
                      for kw in node.keywords}
            return partial(func, **kwargs)
        return None

    def parse(self, csv_file_path: str) -> Case:
        case_list = []
        csv_case = pd.read_csv(csv_file_path, sep='|')
        for case_row in csv_case.iterrows():
            case = Case()
            case_data = case_row[1]
            case.case_name = case_data['CaseName']
            case.op_name = case_data['OpName']
            case.op_param = json.loads(case_data['OpParam'])
            in_num = int(case_data['InNum'])
            in_dtypes = tuple(
                map(lambda t: OP_DTYPES[t], case_data['InDType'].split(';')))
            in_formats = tuple(case_data['InFormat'].split(';'))
            in_shapes = tuple(
                map(lambda s: tuple(map(int, s.split(','))), str(case_data['InShape']).split(';')))
            data_generates = tuple(case_data['DataGenerate'].split(';'))
            for i in range(in_num):
                case.in_tensors.append({
                    'dtype': in_dtypes[i],
                    'format': in_formats[i],
                    'shape': in_shapes[i],
                    'generator': self.__get_generator(data_generates[i])
                })
            out_num = int(case_data['OutNum'])
            out_dtypes = tuple(
                map(lambda t: OP_DTYPES[t], case_data['OutDType'].split(';')))
            out_formats = tuple(case_data['OutFormat'].split(';'))
            out_shapes = tuple(
                map(lambda s: tuple(map(int, s.split(','))), str(case_data['OutShape']).split(';')))
            for i in range(out_num):
                case.out_tensors.append({
                    'dtype': out_dtypes[i],
                    'format': out_formats[i],
                    'shape': out_shapes[i],
                })
            case.test_type = case_data['TestType']
            case.soc_version = case_data['SocVersion']
            case_list.append(case)
        return case_list
