# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import os
import unittest
import logging
import json
import re
import numpy
import torch
import torch_npu
from abc import abstractmethod

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s')

MKI_HOME_PATH = os.environ.get("MKI_HOME_PATH")
if MKI_HOME_PATH is None:
    raise RuntimeError(
        "env MKI_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(MKI_HOME_PATH, "tests/libmki_torch.so")
torch.classes.load_library(LIB_PATH)


class OpTest(unittest.TestCase):

    def setUp(self):
        logging.info("running testcase "
                     f"{self.__class__.__name__}.{self._testMethodName}")
        self.format_default = -1
        self.format_nz = 29
        self.format_nchw = 0
        self.format_nhwc = 1
        self.format_nc1hwc0 = 3
        self.format_nd = 2
        self.multiplex = False
        self.out_flag = False
        self.support_soc = []

    def set_param(self, op_name, op_param):
        self.op_desc = {
            "opName": op_name,
            "specificParam": op_param}
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_param_perf(self, op_name, run_times, op_param):
        self.op_desc = {
            "opName": op_name,
            "runTimes": run_times,
            "specificParam": op_param}
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_support_910b(self):
        self.support_soc.append("Ascend910B")

    def set_support_310p(self):
        self.support_soc.append("Ascend310P")

    def set_support_910b_only(self):
        self.support_soc = ["Ascend910B"]

    def set_support_310p_only(self):
        self.support_soc = ["Ascend310P"]

    def set_input_formats(self, formats):
        self.op_desc["input_formats"] = formats
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def set_output_formats(self, formats):
        self.op_desc["output_formats"] = formats
        self.mki = torch.classes.MkiTorch.MkiTorch(json.dumps(
            self.op_desc))

    def execute(self, in_tensors, out_tensors, envs=None):
        if not self.__set_npu_device():
            return

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]

        self.__set_envs(envs)
        self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        golden_out_tensors = self.golden_calc(in_tensors)
        for idx, tensor in enumerate(in_tensors):
            logging.debug("PythonTest Input Tensor[%s]:", idx)
            logging.debug(tensor)
        for idx, tensor in enumerate(out_tensors):
            logging.debug("PythonTest Output Tensor[%s]:", idx)
            logging.debug(tensor)
        for idx, tensor in enumerate(golden_out_tensors):
            logging.debug("PythonTest Golden Tensor[%s]:", idx)
            logging.debug(tensor)
        self.assertTrue(self.golden_compare(out_tensors, golden_out_tensors))

    def execute_perf(self, in_tensors, out_tensors, envs=None):
        if not self.__set_npu_device():
            return

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]
        self.__set_envs(envs)
        self.run_result = self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        return out_tensors

    def execute_unittest(self, in_tensors, out_tensors, envs=None):
        if not self.__set_npu_device():
            return

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [in_tensors_npu[i] if isinstance(i, int) else i.npu()
                           for i in out_tensors]
        self.__set_envs(envs)
        self.run_result = self.mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if out_tensors_npu:
            out_tensors = [tensor.cpu() for tensor in out_tensors_npu]
        else:
            logging.info("No output tensor, use input tensors as output")
            out_tensors = [tensor.cpu() for tensor in in_tensors_npu]

        return out_tensors

    def __set_envs(self, env: dict):
        if env:
            for key, value in env.items():
                os.environ[key] = value

    def __unset_envs(self, env: dict):
        if env:
            for key, _ in env.items():
                os.environ[key] = ""

    def __get_npu_device(self):
        npu_device = os.environ.get("MKI_NPU_DEVICE")
        if npu_device is None:
            npu_device = "npu:0"
        else:
            npu_device = f"npu:{npu_device}"
        return npu_device

    def __set_npu_device(self):
        npu_device = self.__get_npu_device()
        torch_npu.npu.set_device(npu_device)

        if self.support_soc:
            device_name = torch_npu.npu.get_device_name()
            if re.search("Ascend910B", device_name, re.I):
                soc_version = "Ascend910B"
            elif re.search("Ascend310P", device_name, re.I):
                soc_version = "Ascend310P"
            else:
                logging.error("device_name %s is not supported", device_name)
                return False

            if soc_version not in self.support_soc:
                logging.info("Current soc is %s is not supported for this case: %s",
                             soc_version, str(self.support_soc))
                return False
        return True

    def __create_tensor(self, dtype, format, shape, minValue, maxValue, device=None):
        if device is None:
            device = self.__get_npu_device()
        input = numpy.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input)
        npu_input = torch.from_numpy(input).to(device)
        if format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format)
        return cpu_input, npu_input

    @abstractmethod
    def custom(self, i, dtype, format, shape):
        pass

    def golden_calc(self, in_tensors):
        logging.info("base golden_calc is called")
        return []

    def golden_compare(self, out_tensors, golden_out_tensors):
        if len(out_tensors) != len(golden_out_tensors):
            return False
        for i in range(len(out_tensors)):
            if not torch.allclose(out_tensors[i], golden_out_tensors[i], rtol=0, atol=0):
                return False
        return True
