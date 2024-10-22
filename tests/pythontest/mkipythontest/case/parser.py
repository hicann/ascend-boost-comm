# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABCMeta, abstractmethod
from ast import literal_eval
from typing import Any

import pandas as pd
from mkipythontest.case import Case
from mkipythontest.case.generalize import make_generalized_shapes_and_op_param
from mkipythontest.constant import TENSOR_DTYPES, OpType, TestType
from mkipythontest.tensor.format import TensorFormat
from numpy import nan


class BaseParser(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, csv_file_path: str) -> list[Case]:
        pass


class DefaultCsvParser(BaseParser):
    def __get_dtypes(self, dtypes_str: str, broadcast_num: int = 1) -> list[Any]:
        if not dtypes_str:
            return []
        dtypes = list(map(lambda t: TENSOR_DTYPES[t], dtypes_str.split(';')))
        if len(dtypes) == 1 and len(dtypes) != broadcast_num:
            dtypes = dtypes*broadcast_num
        return dtypes

    def __get_shapes(self, shapes_str: str, broadcast_num: int = 1) -> list[tuple]:
        if not shape_str:
            return []
        shapes = []
        for shape_str in shapes_str.split(';'):
            dims = []
            for dim in shape_str.split(','):
                if dim.isnumeric():
                    dims.append(int(dim))
                else:
                    dims.append(dim)
            shapes.append(tuple(dims))
        if len(shapes) == 1 and len(shapes) != broadcast_num:
            shapes = shapes*broadcast_num
        return shapes

    def __get_formats(self, formats_str: str, broadcast_num: int = 1) -> list[TensorFormat]:
        if not formats_str:
            return []
        formats = list(
            map(lambda f: TensorFormat[f], tuple(formats_str.split(';'))))
        if len(formats) == 1 and len(formats) != broadcast_num:
            formats = formats * broadcast_num
        return formats

    def parse(self, csv_file_path: str, filter: dict = {}) -> list[Case]:
        case_list = []
        csv_case = pd.read_csv(csv_file_path, sep='|')
        csv_case = csv_case.replace(nan, '')
        for case_row in csv_case.iterrows():

            case_data = case_row[1]
            test_type = TestType[case_data['TestType']]

            case.op_name = case_data['OpName']
            case.op_type = OpType[case_data['OpType']]

            case = Case(case_data['CaseName'], in_num, out_num)
            case.in_dtypes = self.__get_dtypes(case_data['InDType'], in_num)
            case.in_formats = self.__get_formats(case_data['InFormat'], in_num)
            case.data_generate = case_data['DataGenerate']
            case.out_dtypes = self.__get_dtypes(case_data['OutDType'], out_num)
            case.out_formats = self.__get_formats(
                case_data['OutFormat'], out_num)

            case.soc_version = tuple(case_data['SocVersion'].split(","))
            case.env = literal_eval(case_data['Env'])
            case.random_seed = case_data['RandomSeed']

            op_param = literal_eval(case_data['OpParam'])
            in_num = case_data['InNum']
            in_shapes = self.__get_shapes(str(case_data['InShape']), in_num)
            out_num = case_data['OutNum']
            out_shapes = self.__get_shapes(str(case_data['OutShape']), out_num)
            if test_type == TestType.GENERALIZATION:
                case_list.extend(make_generalized_shapes_and_op_param(
                    case, in_shapes, out_shapes, op_param))
            else:
                case_list.append(case)

            # case.dump = bool(case_data['Dump'])

        return case_list

    def generate(self, csv_file_path: str, case_list: list[Case]) -> None:
        pass
