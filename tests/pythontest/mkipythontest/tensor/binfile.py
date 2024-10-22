# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import logging

import numpy
import torch
from mkipythontest.tensor.format import TensorFormats

ATTR_VERSION = "$Version"
ATTR_END = "$End"
ATTR_OBJECT_LENGTH = "$Object.Length"
ATTR_OBJECT_PREFIX = "$Object."


class TensorLoadSave:
    def __init__(self, bin_file_path: str) -> None:
        self.obj_buffer = None
        self.bin_file_path: str = bin_file_path
        self.dtype: torch.dtype = torch.float32
        self.format_: TensorFormats = TensorFormats.ND
        self.shape: tuple[int] = ()

        self.__obj_buffer = None
        self.__obj_len = None

    def load_bin_file(self, bin_file_path: str) -> torch.Tensor:
        end_str = f"{ATTR_END}=1"
        with open(bin_file_path, mode="rb") as bin_file:
            bin_data = bin_file.read()
            begin_offset = 0
            for i in range(len(bin_data)):
                if bin_data[i] != ord('\n'):
                    continue
                line = bin_data[begin_offset:i].decode("utf-8")
                begin_offset = i + 1
                fields = line.split('=')
                attr_name = fields[0]
                attr_value = fields[1]
                if attr_name == ATTR_END:
                    self.obj_buffer = bin_data[i + 1:]
                    break
                elif attr_name.startswith('$'):
                    self.__parse_system_attr(attr_name, attr_value)
                else:
                    self.__parse_user_attr(attr_name, attr_value)
        return torch.tensor(numpy.frombuffer(self.__obj_buffer)).view(self.shape).to(self.dtype)

    def save_bin_file(self, tensor: torch.Tensor, bin_file_path: str) -> None:
        with open(bin_file_path, mode="rb") as bin_file:
            pass

    def __parse_system_attr(self, attr_name: str, attr_value: str):
        if attr_name == ATTR_OBJECT_LENGTH:
            self.__obj_len = int(attr_value)
        elif attr_name == ATTR_OBJECT_PREFIX:
            logging.info("reserved")

    def __parse_user_attr(self, attr_name: str, attr_value: str):
        if attr_name == "dtype":
            self.dtype = TENSOR_DTYPES[int(attr_value)]
        if attr_name == "format":
            self.format_ = TENSOR_FORMATS[int(attr_value)]
        if attr_name == "dims":
            self.shape = tuple(map(int, attr_value.split(',')))
