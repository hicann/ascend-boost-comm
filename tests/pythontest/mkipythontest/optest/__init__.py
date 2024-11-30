# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import logging
import os
import re
import unittest
from abc import abstractmethod
from typing import Union

import numpy
import torch
import torch_npu
from mkipythontest.case import Case
from mkipythontest.constant import ErrorType, OpType, TensorFormat
from mkipythontest.optest.compare import ComparerFactory
from mkipythontest.optest.error import MkiOpTestError
from mkipythontest.optest.runner import MkiTorchRunner
from mkipythontest.utils.env import get_npu_device, set_envs, unset_envs
from mkipythontest.utils.generator import (gen_tensors,
                                           get_param_from_generator_str)
from mkipythontest.utils.log import log_tensors

logging.basicConfig(
    # level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
)


class OpTest(unittest.TestCase):
    test_cases: dict[str, Case] = {}
    op_type: OpType = OpType.NA
    use_gpu_golden: bool = False
    iomux: dict[int, int] = {}

    @classmethod
    def setUpClass(cls):
        npu_device = get_npu_device()
        torch_npu.npu.set_device(npu_device)

    def setUp(self):
        logging.info(
            f"running testcase{self.__class__.__name__}.{self.get_case_name()}"
        )
        self._op_desc: dict = {}
        self._in_tensors = []
        self._out_tensors = []

    def set_rand_seed(self, rand_seed: int):
        """Set random seed for `torch` and `numpy`.

        :param rand_seed: specific random seed
        """
        self._random_seed = rand_seed
        numpy.random.seed(self._random_seed)
        torch.manual_seed(self._random_seed)

    @classmethod
    def get_op_name(cls) -> str:
        """Get the operation name when a good class name is given.

        :return: operation name
        """
        class_name = cls.__name__
        match_result = re.findall(
            r'Test([A-Z][A-Za-z0-9]+Operation)([A-Za-z0-9]*)', class_name)
        if not match_result:
            logging.info(
                "The class name is not good. Please rename it to 'Test{OpName}Operation{KernelName}'.")
            return class_name
        return match_result[0][0]

    def get_case_name(self) -> str:
        """Get test case name.

        :return: current case name
        """
        return self._testMethodName.replace("test_", "", 1)

    def set_param(self, op_param: dict, op_name=None):
        """Set operation param and name.
        Usually, you must name the class well then the operation name will be got automatically.

        :param op_param: operation param
        :param op_name: operation name, defaults to None
        """
        if not op_name:
            op_name = self.get_op_name()
        self._op_desc = {"opName": op_name, "specificParam": op_param}

    def get_param(self) -> dict:
        """Get operation specific param.

        :return: specific param
        """
        return self._op_desc["specificParam"]

    def set_input_formats(self, formats: list[TensorFormat]):
        """Set operation input formats.

        :param formats: input formats
        """
        self._op_desc["input_formats"] = [format_.value for format_ in formats]

    def set_output_formats(self, formats: list[TensorFormat]):
        """Set operation output formats.

        :param formats: output formats
        """
        self._op_desc["output_formats"] = [
            format_.value for format_ in formats]

    def execute(self, in_tensors: list[torch.Tensor],
                out_tensors: list[Union[torch.Tensor, int]],
                envs=None):
        """Do operation executing.

        :param in_tensors: input
        :param out_tensors: output
        :param envs: used by some operation, defaults to None
        :raises RuntimeError: raises when operation running failed
        """

        # for special compute
        if envs is None:
            envs = {}
        self._in_tensors = in_tensors
        self._out_tensors = out_tensors

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [
            in_tensors_npu[i] if isinstance(i, int) else i.npu() for i in out_tensors
        ]

        runner = MkiTorchRunner(self._op_desc)

        set_envs(envs)
        run_result = runner.execute(in_tensors_npu, out_tensors_npu)
        unset_envs(envs)

        if run_result != ErrorType.NO_ERROR:
            raise MkiOpTestError(run_result)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        golden_out_tensors = self.golden_calc(in_tensors)
        logging.debug("Input Tensors:")
        log_tensors(in_tensors)
        logging.debug("Output Tensors:")
        log_tensors(out_tensors)
        logging.debug("Golden Output Tensors:")
        log_tensors(golden_out_tensors)

        if self.use_gpu_golden:
            golden_out_tensors_gpu = self.golden_calc_gpu(in_tensors)
            logging.debug("GPU Golden Output Tensors:")
            log_tensors(golden_out_tensors)
            compare_result = self.golden_compare(
                out_tensors, golden_out_tensors, golden_out_tensors_gpu)
        else:
            compare_result = self.golden_compare(
                out_tensors, golden_out_tensors)

        self.assertTrue(compare_result)
        if os.getenv("MKI_TEST_PERFORMANCE"):
            pass
            # with open...

    def run_case(self, case: Case):
        """Run a case.

        """
        self.set_param(case.op_param, case.op_name)
        self.set_rand_seed(case.random_seed)

        in_tensors = self.__data_generate_inner(case)
        out_tensors = case.out_tensors

        self.set_input_formats(case.in_formats)
        self.set_output_formats(case.out_formats)

        case.iomux.update(self.iomux)

        for output_idx, input_idx in case.iomux.items():
            out_tensors[output_idx] = input_idx

        self.execute(in_tensors, out_tensors, envs=case.env)

    def __data_generate_inner(self, case: Case) -> list[torch.Tensor]:
        """Data generation. Use custom or generate string.

        :return: generated tensor list
        """
        if case.data_generate.startswith("custom"):
            return self.data_generate(case.in_tensors, **get_param_from_generator_str(case.data_generate))
        else:
            data_generate_strs = case.data_generate.split(';')
            return gen_tensors(case.in_dtypes, case.in_shapes, data_generate_strs)

    def data_generate(self, in_tensors: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        """Tensor custom generate interface. Defaults to U(-5,5).

        :param in_tensors: input tensors which have only shape and dtype
        :param kwargs: some other custom arguments, passing by `custom(k1=v1,k2=v2)`
        :return: generated tensor list
        """
        for i in range(len(in_tensors)):
            in_tensors[i] = in_tensors[i].uniform_(-5, 5)
        return in_tensors

    def calculate_times(self, in_tensors: list[torch.Tensor]) -> int:
        """Calculate the calculate times of the operation.

        :param in_tensors: input tensors
        :return: calculate times
        """
        return 0

    @abstractmethod
    def golden_calc(self, in_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """CPU golden interface.

        :param in_tensors: input tensors
        :return: golden output tensors
        """
        pass

    @abstractmethod
    def golden_calc_gpu(self, in_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """GPU golden interface.

        :param in_tensors: input tensors
        :return: golden output tensors
        """
        pass

    def golden_compare(
            self,
            out_tensors: list[torch.Tensor],
            golden_out_tensors: list[torch.Tensor],
            golden_out_tensors_gpu=None,
    ):
        """Compare results.

        :param out_tensors: output tensors
        :param golden_out_tensors: golden output tensors from CPU
        :param golden_out_tensors_gpu: golden output tensors from GPU
        :return: Compare result
        """
        if golden_out_tensors_gpu is None:
            golden_out_tensors_gpu = []
        results = []
        compare_factory = ComparerFactory()
        compare_factory.set_op_type(self.op_type)
        for i in range(len(out_tensors)):
            out_tensor = out_tensors[i]
            golden_out_tensor = golden_out_tensors[i]
            calculate_times = self.calculate_times(self._in_tensors)
            compare_factory.set_calculate_times(calculate_times)
            logging.info(
                f"Tensor {str(i)}: OP Type: {self.op_type}, dtype {out_tensor.dtype}, calculate_times: {str(calculate_times)}")
            if self.use_gpu_golden:
                comparer = compare_factory.get_double_golden_comparer(
                    out_tensor.dtype)
                compare_result = comparer(
                    out_tensor, golden_out_tensor, golden_out_tensors_gpu[i])
            else:
                comparer = compare_factory.get_single_golden_comparer(
                    out_tensor.dtype)
                compare_result = comparer(
                    out_tensor, golden_out_tensor)
            results.append(compare_result)
        for i, result in enumerate(results):
            logging.info(
                f"Tensor {str(i)}: {result}")
        return all(results)
