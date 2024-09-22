# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import json
from abc import ABCMeta, abstractmethod
from typing import Any

import pandas as pd
from mkipythontest.case import Case, GeneralizedCase
from mkipythontest.constant import TENSOR_DTYPES, OpType, TestType
from mkipythontest.tensor.format import TENSOR_FORMATS, TensorFormat


class BaseParser(metaclass=ABCMeta):
    @abstractmethod
    def parse(self, csv_file_path: str) -> list[Case]:
        pass


class DefaultCsvParser(BaseParser):
    def __get_dtypes(self, dtypes_str: str) -> list[Any]:
        return list(
            map(lambda t: TENSOR_DTYPES[t], dtypes_str.split(';')))

    def __get_shapes(self, shapes_str: str) -> list[tuple]:
        result = []
        for shape_str in shapes_str.split(';'):
            dims = []
            for dim in shape_str.split(','):
                if dim.isnumeric():
                    dims.append(int(dim))
                else:
                    dims.append(dim)
            result.append(tuple(dims))
        return result

    def __get_formats(self, formats_str: str) -> list[TensorFormat]:
        return list(map(lambda f: TENSOR_FORMATS[f], tuple(formats_str.split(';'))))

    def parse(self, csv_file_path: str) -> list[Case]:
        case_list = []
        csv_case = pd.read_csv(csv_file_path, sep='|')
        for case_row in csv_case.iterrows():
            case_data = case_row[1]
            test_type = TestType[case_data['TestType']]
            case = Case(case_data['CaseName'])
            if test_type == TestType.GENERALIZATION:
                case = GeneralizedCase(case_data['CaseName'])

            case.op_name = case_data['OpName']
            case.op_type = OpType[case_data['OpType']]
            case.op_param = json.loads(case_data['OpParam'])

            case.in_dtypes = self.__get_dtypes(case_data['InDType'])
            case.in_shapes = self.__get_shapes(case_data['InShape'])
            case.in_formats = self.__get_formats(case_data['InFormat'])

            case.data_generate = case_data['DataGenerate']

            case.out_dtypes = self.__get_dtypes(case_data['OutDType'])
            case.out_shapes = self.__get_shapes(case_data['OutShape'])
            case.out_formats = self.__get_formats(case_data['OutFormat'])

            case.soc_version = tuple(case_data['SocVersion'].split(","))
            case.env = json.loads(case_data['Env'])
            case.dump = bool(case_data['Dump'])

            case_list.append(case)
        return case_list

    def generate(self, csv_file_path: str, case_list: list[Case]) -> None:
        pass
