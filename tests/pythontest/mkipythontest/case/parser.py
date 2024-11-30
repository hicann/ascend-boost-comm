# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
from abc import ABCMeta, abstractmethod
from ast import literal_eval
from typing import Any

import pandas as pd
import torch
from mkipythontest.case import Case
from mkipythontest.case.generalize import make_generalized_case_list
from mkipythontest.constant import (ERROR_DICT, TENSOR_DTYPES_DICT,
                                    TENSOR_FORMAT_DICT, ErrorType,
                                    TensorFormat)
from mkipythontest.utils.misc import is_sub_dict, split_and_map_dict
from mkipythontest.utils.soc import SOC_SUPPORT, SocType
from numpy import nan


class BaseParser(metaclass=ABCMeta):
    def filter(self, data_dict: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if a case is useless by checking sub dict

        :param data_dict: case data dict
        :param filter_dict: filter data dict
        :return: if a case is useless
        """
        if filter_dict == {}:
            return True
        return is_sub_dict(filter_dict, data_dict)

    @abstractmethod
    def parse(self, csv_file_path: str, filter_dict: dict[str, Any]) -> list[Case]:
        pass


class MkiCsvParser(BaseParser):
    def get_dtypes(self, dtypes_str: str, broadcast_num: int = 1) -> list[torch.dtype]:
        dtypes = split_and_map_dict(dtypes_str, TENSOR_DTYPES_DICT,
                                    nonexistent_key_default_value=torch.float16)
        if len(dtypes) == 1 and len(dtypes) != broadcast_num:
            dtypes = dtypes*broadcast_num
        return dtypes

    def get_shapes(self, shapes_str: str, broadcast_num: int = 1) -> list[tuple]:
        if not shapes_str:
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

    def get_formats(self, formats_str: str, broadcast_num: int = 1) -> list[TensorFormat]:
        formats = split_and_map_dict(formats_str, TENSOR_FORMAT_DICT,
                                     nonexistent_key_default_value=TensorFormat.ND)
        if len(formats) == 1 and len(formats) != broadcast_num:
            formats = formats * broadcast_num
        return formats

    def get_dict(self, dict_str: str) -> dict:
        return literal_eval(dict_str) if dict_str else {}

    def get_iomux_dict(self, iomux_str: str) -> dict[int, int]:
        return {int(iomux[2]): int(iomux[0]) for iomux in iomux_str.split(";")} if iomux_str else {}

    def get_soc_version(self, soc_version_str: str) -> list[SocType]:
        if not soc_version_str:
            return []
        soc_version_splited = soc_version_str.split(";")
        # check if mix
        skip_mode = False
        soc_version_list = []
        for soc_version in soc_version_splited:
            if skip_mode and not soc_version.startswith('!'):
                return []
            if soc_version.startswith('!'):
                skip_mode = True
            soc_version_list.append(SocType[soc_version.replace('!', '')])
        if skip_mode:
            return SOC_SUPPORT.difference(soc_version_list)
        return soc_version_list

    def get_expected_error(self, expected_error_str: str) -> ErrorType:
        return ERROR_DICT.get(
            expected_error_str, ErrorType.NO_ERROR)

    def parse(self, csv_file_path: str, filter=None) -> list[Case]:
        if filter is None:
            filter = {}
        case_list = []
        csv_case = pd.read_csv(csv_file_path, sep='|')
        csv_case = csv_case.replace(nan, '')
        for case_row in csv_case.iterrows():
            case_data = case_row[1]
            if not self.filter(case_data, filter):
                continue

            in_num = case_data['InNum']
            out_num = case_data['OutNum']
            case = Case(case_data['CaseName'], in_num, out_num)
            case.op_name = case_data['OpName']
            case.in_dtypes = self.get_dtypes(case_data['InDType'], in_num)
            case.in_formats = self.get_formats(case_data['InFormat'], in_num)
            case.data_generate = case_data['DataGenerate']
            case.out_dtypes = self.get_dtypes(case_data['OutDType'], out_num)
            case.out_formats = self.get_formats(
                case_data['OutFormat'], out_num)

            # below is optional
            case.soc_version = self.get_soc_version(
                case_data.get('SocVersion', ''))
            case.expected_error = self.get_expected_error(
                case_data.get('ExpectedError', "NO_ERROR"))
            case.env = self.get_dict(case_data.get('Env', ''))
            case.iomux = self.get_iomux_dict(case_data.get('IOMux', ''))
            # case.random_seed = case_data['RandomSeed']

            op_param = self.get_dict(case_data['OpParam'])
            in_shapes = self.get_shapes(str(case_data['InShape']), in_num)
            out_shapes = self.get_shapes(str(case_data['OutShape']), out_num)
            if os.getenv('MKI_TEST_GENERALIZATION'):
                generalization_dims = {}
                generalization_dims_env = os.getenv(
                    'MKI_TEST_GENERALIZATION_DIMS')
                if generalization_dims_env is not None:
                    generalization_dims = literal_eval(generalization_dims_env)
                case_list.extend(make_generalized_case_list(
                    case, in_shapes, out_shapes, op_param, generalization_dims))
            else:
                case.op_param = op_param
                case.in_shapes = in_shapes
                case.out_shapes = out_shapes
                case_list.append(case)
        return case_list


class ProfilingCsvParser(MkiCsvParser):
    def get_dtypes(self, dtypes_str: str):
        return super().get_dtypes(dtypes_str.lower())

    def get_formats(self, formats_str: str):
        return super().get_formats(formats_str.replace("FORMAT_", "").lower())

    def parse(self, csv_file_path: str, filter=None) -> list[Case]:
        if filter is None:
            filter = {}
        case_list = []
        csv_case = pd.read_csv(csv_file_path, sep=',')
        csv_case = csv_case.replace(nan, '')
        for case_row in csv_case.iterrows():
            case_data = case_row[1]
            if not self.filter(case_data, filter):
                continue
            case = Case()
            case.op_name = case_data['Op Name']
            case.in_dtypes = self.get_dtypes(case_data['Input Data Types'])
            case.in_formats = self.get_formats(case_data['Input Formats'])
            case.out_dtypes = self.get_dtypes(case_data['Output Data Types'])
            case.out_formats = self.get_formats(case_data['Output Formats'])
            case.op_param = literal_eval(case_data['OpParam'])
            case.in_shapes = self.get_shapes(str(case_data['Input Shape']))
            case.out_shapes = self.get_shapes(str(case_data['Output Shape']))
            case_list.append(case)
        return case_list
