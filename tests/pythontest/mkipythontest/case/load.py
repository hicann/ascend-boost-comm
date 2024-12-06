# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import logging
from pathlib import Path
from typing import Type

import torch
from mkipythontest.case import Case
from mkipythontest.constant import ErrorType
from mkipythontest.optest import OpTest
from mkipythontest.optest.error import expect_error
from mkipythontest.utils.soc import on_soc

from .parser import BaseParser, MkiCsvParser


def add_case(test_class: Type[OpTest], case: Case):
    """Add case to a test class.

    :param test_class: test class
    :param case: case
    """
    test_class_op_name = test_class.get_op_name()
    if case.op_name and case.op_name != test_class_op_name:
        logging.info(
            f"test class opname {test_class_op_name} does not match case opname {case.op_name}. ignored.")
        return
    if case.case_name in test_class.test_cases:
        logging.info("found duplicate case name. ignored.")
        return
    test_class.test_cases[case.case_name] = case

    def __test_func(self: OpTest): return self.run_case(case)
    if case.soc_version:
        __test_func = on_soc(soc_type=case.soc_version)(__test_func)
    if case.expected_error != ErrorType.NO_ERROR:
        __test_func = expect_error(case.expected_error)
    setattr(test_class, f'test_{case.case_name}', __test_func)
    logging.info(f"loading case \"{case.case_name}\" OK!")


def add_bin_case(test_class: Type[OpTest], case: Case, in_tensors: list[torch.Tensor], out_tensors: list[torch.Tensor]):
    def __test_func(self: OpTest):
        self.set_param(case.op_param)
        self.set_input_formats(case.in_formats)
        self.set_output_formats(case.out_formats)
        self.execute(in_tensors, out_tensors)
    setattr(test_class, f'test_bin_file', __test_func)
    logging.info(f"loading bin case OK!")


def load_csv_to_optest(test_class: Type[OpTest],
                       csv_path: str,
                       parser: BaseParser = MkiCsvParser()) -> Type[OpTest]:
    """Load cases from a csv and inject cases to a test class.

    :param test_class: test class
    :param csv_path: path of csv file
    :param parser: csv parser object
    :return: test class with cases
    """

    if not Path(csv_path).is_file():
        logging.error("not csv file")
    cases = parser.parse(csv_path)
    for case in cases:
        add_case(test_class, case)
    return test_class
