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
import logging
import os
import re
import unittest
import warnings
from abc import abstractmethod
from random import randint

import numpy
import pandas
import torch

try:
    import torch_npu
except ImportError:
    pass
from mkipythontest import TestType
from mkipythontest.case import Case
from mkipythontest.constant import OpType
from mkipythontest.tensor.compare import ComparerFactory
from mkipythontest.utils.profiler import get_profiler_time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)s - %(levelname)s - %(message)s",
)


class OpTest(unittest.TestCase):
    # 这里放成员
    test_cases: dict[str, Case] = {}
    test_results: dict[str, dict[str]] = {}
    op_type: OpType = OpType.NA
    use_gpu_golden: bool = False

    @classmethod
    def setUpClass(cls):
        MKI_HOME_PATH = os.environ.get("MKI_HOME_PATH")
        if MKI_HOME_PATH is None:
            raise RuntimeError(
                "env MKI_HOME_PATH not exist, source set_env.sh")
        LIB_PATH = os.path.join(MKI_HOME_PATH, "tests/libmki_torch.so")
        torch.classes.load_library(LIB_PATH)

    def setUp(self):
        # 这里放context
        logging.info(
            "running testcase " f"{self.__class__.__name__}.{
            self._testMethodName}"
        )
        self.format_default = -1
        self.format_nz = 29
        self.format_nchw = 0
        self.format_nhwc = 1
        self.format_nc1hwc0 = 3
        self.format_nd = 2
        self.multiplex = False
        self.out_flag = False
        self.support_soc = []
        self._mki: torch.ClassDef = None
        self._op_desc: dict = {}
        # set random seed
        self._random_seed = randint(0, 1000000)
        numpy.random.seed(self._random_seed)
        torch.manual_seed(self._random_seed)

    def tearDown(self):
        self.test_results[self._testMethodName] = {
            "RandomSeed": self._random_seed
        }

    @classmethod
    def tearDownClass(cls):
        pass

    # 把那个dict写csv里
    # write self.test_results int ocsv
    # the key is CaseName, and follows other result
    # results_dataframe = pandas.DataFrame.from_dict(
    #     self.test_cases, orient="index")
    # results_dataframe = pandas.DataFrame(self.test_results)
    # results_dataframe.to_csv(f"{self.__class__.__name__}.csv")

    def get_op_name(self):
        class_name = self.__name__
        if not re.match(r'Test[A-Z]{1}[A-Za-z0-9]+Operation[A-Za-z0-9]+', class_name):
            logging.info("The class name is not good. Please rename it to 'Test{OpName}Operation{KernelName}'.")
            return class_name
        return class_name.split("Operation")[0].replace("Test", "", 1) + "Operation"

    def set_param(self, op_param, op_name=None):
        if not op_name:
            op_name = self.get_op_name()
        self._op_desc = {"opName": op_name, "specificParam": op_param}

    def set_param_perf(self, op_name, run_times, op_param):
        self._op_desc = {
            "opName": op_name,
            "runTimes": run_times,
            "specificParam": op_param,
        }

    def get_param(self):
        return self._op_desc["specificParam"]

    def set_support_910b(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead",
            DeprecationWarning,
        )
        self.support_soc.append("Ascend910B")

    def set_support_310p(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead",
            DeprecationWarning,
        )
        self.support_soc.append("Ascend310P")

    def set_support_910b_only(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead",
            DeprecationWarning,
        )
        self.support_soc = ["Ascend910B"]

    def set_support_310p_only(self):
        warnings.warn(
            "It is useless and will be removed recently, please use soc decorator instead",
            DeprecationWarning,
        )
        self.support_soc = ["Ascend310P"]

    def set_input_formats(self, formats):
        self._op_desc["input_formats"] = formats

    def set_output_formats(self, formats):
        self._op_desc["output_formats"] = formats

    def execute(self, in_tensors, out_tensors, perf_times=1, envs=None):
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
                quit(1)

            if soc_version not in self.support_soc:
                logging.info(
                    "Current soc is %s is not supported for this case: %s",
                    soc_version,
                    str(self.support_soc),
                )
                return

        # for special compute
        self._in_tensors = in_tensors

        in_tensors_npu = [tensor.npu() for tensor in in_tensors]
        out_tensors_npu = [
            in_tensors_npu[i] if isinstance(i, int) else i.npu() for i in out_tensors
        ]

        mki = torch.classes.MkiTorch.MkiTorch(json.dumps(self._op_desc))

        self.__set_envs(envs)
        run_result = mki.execute(in_tensors_npu, out_tensors_npu)
        self.__unset_envs(envs)

        if run_result != "ok":
            raise RuntimeError(run_result)

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

        if self.use_gpu_golden:
            golden_out_tensors_gpu = self.golden_calc_gpu(in_tensors)
            for idx, tensor in enumerate(golden_out_tensors_gpu):
                logging.debug("PythonTest GpuGolden Tensor[%s]:", idx)
                logging.debug(tensor)
            compare_result = self.golden_compare(out_tensors, golden_out_tensors, golden_out_tensors_gpu)
        else:
            compare_result = self.golden_compare(out_tensors, golden_out_tensors)

        self.test_results[self._testMethodName.replace(
            "test_", "")] = compare_result
        self.assertTrue(compare_result)

        if perf_times > 1:
            warmup_times = perf_times / 10
            active_times = perf_times - warmup_times
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
                data_simplification=True
            )
            with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(
                        wait=0, warmup=warmup_times, active=active_times, repeat=1, skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        "/tmp"),
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    experimental_config=experimental_config
            ) as profiler:
                for _ in range(perf_times):
                    self.execute(in_tensors, out_tensors, envs=envs)
                    profiler.step()
            self.time_duration = get_profiler_time("aaaa")

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

    def __create_tensor(self, dtype, format, shape, minValue, maxValue, device=None):
        if device is None:
            device = self.__get_npu_device()
        input = numpy.random.uniform(minValue, maxValue, shape).astype(dtype)
        cpu_input = torch.from_numpy(input)
        npu_input = torch.from_numpy(input).to(device)
        if format != -1:
            npu_input = torch_npu.npu_format_cast(npu_input, format)
        return cpu_input, npu_input

    def run_case(self, case_name: str) -> None:
        """
        运行用例

        :param case: 用例对象
        :return:
        """
        case = self.test_cases[case_name]
        # if not (isinstance(case, Case) and not issubclass(type(case), Case)):
        #     logging.error("cannot directly run derived case.")
        #     return
        # set param
        self.set_param(case.op_name, case.op_param)

        in_tensors = case.generate_data(self.data_generate)
        out_tensors = case.out_tensors

        # format convert

        # for i in range(len(case.in_formats)):
        #     in_tensors[i] = convert_format(in_tensors[i], ND, case.in_formats[i])

        if case.test_type == TestType.PERFORMANCE:
            self.execute(in_tensors, out_tensors, perf_times=20, envs=case.env)
        else:
            self.execute(in_tensors, out_tensors, envs=case.env)

        if case.dump:
            # dump
            pass

    def data_generate(self, in_tensors: list[torch.Tensor], **kwargs):
        """
        Tensor数据生成
        默认以U(-5,5)填充算子输入

        :param op_param: 算子参数
        :param in_tensors: 算子输入（为空）
        :param kwargs: 可选的其他参数
        :return:
        """
        for i in range(len(in_tensors)):
            in_tensors[i] = in_tensors[i].uniform_(-5, 5)
        return in_tensors

    def calculate_times(self, in_tensors: list[torch.Tensor]) -> int:
        """
        获取算子计算次数，用于新精度分支选择
        默认返回0，一般对应最严格的比较标准

        :param op_param: 算子参数
        :param in_tensors: 算子输入
        :return: 算子计算次数
        """
        return 0

    @abstractmethod
    def golden_calc(self, in_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        本地计算标杆，一般为CPU侧计算
        无默认实现

        :param in_tensors: 算子输入
        :return: 标杆
        """
        pass

    @abstractmethod
    def golden_calc_gpu(self, in_tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        第二标杆，一般为GPU侧计算
        无默认实现

        :param in_tensors: 算子输入
        :return: 标杆
        """
        pass

    def golden_compare(
            self,
            out_tensors: list[torch.Tensor],
            golden_out_tensors: list[torch.Tensor],
            golden_out_tensors_gpu: list[torch.Tensor] = [],
    ):
        """
        输出与标杆比较

        :param out_tensors: 算子输出
        :param golden_out_tensors: 算子输出标杆
        :param golden_out_tensors_gpu: 算子输出第二标杆
        :return:
        """
        results = []
        compare_factory = ComparerFactory()
        compare_factory.set_op_type(self.op_type)
        for i in range(len(out_tensors)):
            out_tensor = out_tensors[i]
            golden_out_tensor = golden_out_tensors[i]
            calculate_times = self.calculate_times(self._in_tensors)
            compare_factory.set_calculate_times(calculate_times)

            if self.use_gpu_golden:
                comparer = compare_factory.get_double_golden_comparer(out_tensor.dtype)
                compare_result = comparer.compare(out_tensor, golden_out_tensor, golden_out_tensors_gpu[i])
            else:
                comparer = compare_factory.get_single_golden_comparer(out_tensor.dtype)
                compare_result = comparer.compare(out_tensor, golden_out_tensors[i])
            results.append(compare_result)
        return all(results)
