# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

from abc import ABCMeta

import numpy
import pandas
import torch
from mkipythontest.constant import OpType

ERR_EXP = {
    (OpType.COMPUTE_FLOAT, torch.float16): -8,
    (OpType.COMPUTE_FLOAT, torch.bfloat16): -7,
    (OpType.COMPUTE_FLOAT, torch.float32): -11,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.float16): -11,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.bfloat16): -8,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.float32): -14,
    (OpType.VECTOR_FUSION, torch.float16): -9,
    (OpType.VECTOR_FUSION, torch.bfloat16): -8,
    (OpType.VECTOR_FUSION, torch.float32): -11,
}

EB_EXP = {
    (OpType.COMPUTE_FLOAT, torch.float16): -10,
    (OpType.COMPUTE_FLOAT, torch.bfloat16): -7,
    (OpType.COMPUTE_FLOAT, torch.float32): -14
}


class CompareResult:
    """
    比较结果
    """

    # todo 这里会之后照搬perftest
    def __init__(self, pass_: bool, pass_ratio: float, eb: float, other_info: dict = {}):
        self.pass_: bool = pass_
        self.pass_ratio: float = pass_ratio
        self.eb: float = eb
        self.other_info: dict = other_info

    def to_dataframe(self):
        return pandas.DataFrame(
            {'Pass':[self.pass_],
             'PassRatio':[self.pass_ratio],
             'EB':self.eb,
             'OtherInfo':self.other_info}
        )

    def __bool__(self):
        return self.pass_


class BaseCompare(metaclass=ABCMeta):
    def mare(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
        """
        计算最大误差

        :param out_tensor:
        :param golden_out_tensor:
        :return:
        """
        abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
        return torch.max(torch.div(abs_diff, torch.abs(golden_out_tensor))).item()

    def mere(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
        """
        计算平均误差

        :param out_tensor:
        :param golden_out_tensor:
        :return:
        """
        abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
        return torch.mean(torch.div(abs_diff, torch.abs(golden_out_tensor))).item()

    def rmse(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
        """
        计算均方根误差

        :param out_tensor: 输出张量
        :param golden_out_tensor: 标杆张量
        :return:
        """
        abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
        return numpy.sqrt(torch.sum(torch.mul(abs_diff, abs_diff) / torch.numel(abs_diff)).numpy())

    def eb(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
        """
        计算误差变异率

        :param out_tensor:
        :param golden_out_tensor:
        :return:
        """
        diff = torch.diff(out_tensor, golden_out_tensor)
        golden_nmax = torch.clamp(torch.abs(golden_out_tensor), min=1)
        return torch.mean(diff, golden_nmax).item()

    def err(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> torch.Tensor:
        """

        :param out_tensor:
        :param golden_out_tensor:
        :return:
        """
        # 输出和标杆reshape成一维
        out_flatten = out_tensor.flatten(-1)
        golden_out_flatten = golden_out_tensor.flatten(-1)
        result = torch.empty(size=out_flatten.shape)
        for i in range(out_flatten.shape[0]):
            out_value = out_flatten[i]
            golden_out_value = golden_out_flatten[i]
            err_value = 0
            # 标杆数值大于1时采用相对误差
            if golden_out_value >= 1.0:
                err_value = abs((golden_out_value - out_value) /
                                max(self.ERR, golden_out_value))
            # 否则采用绝对误差
            else:
                err_value = abs(golden_out_value - out_value)
            result[i] = err_value
        return result

    def __call__(self, *args, **kwargs):
        pass


class SingleCompare(BaseCompare, metaclass=ABCMeta):
    """
    单标杆比较器
    """

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> CompareResult:
        pass


class DoubleCompare(BaseCompare, metaclass=ABCMeta):
    """
    双标杆比较器
    """

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor,
                 gpu_golden_out_tensor: torch.Tensor) -> CompareResult:
        pass


class BinaryMatchCompare(SingleCompare):
    """
    二进制一致比较器，用于MOVE/CAST/COMPUTE_INTEGER
    """

    def __init__(self) -> None:
        pass

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> CompareResult:
        pass_ratio = torch.sum(
            out_tensor == golden_out_tensor).item() / out_tensor.numel()
        return CompareResult(pass_ratio == 1, pass_ratio, 0, "")


class RandomCompare(SingleCompare):
    """
    随机算子比较器
    """

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor = []) -> CompareResult:
        alpha = 0.01
        z = -3.0902
        case_pass_percent = ((1 - alpha) + z * numpy.sqrt(alpha * (1 - alpha)))
        return CompareResult(1, 0, "")


class QuantCompare(SingleCompare):
    """
    量化运算比较器
    """
    pass


class EBCompare(SingleCompare):
    """
    测量误差变异率的比较器
    """

    def __init__(self, err, eb):
        self.ERR = err
        self.EB = eb

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> CompareResult:
        # 通过计数
        pass_count = 0
        result = self.err(out_tensor, golden_out_tensor)
        for err_value in result:
            if err_value < self.ERR:
                pass_count += 1
        return CompareResult(pass_count / result.shape[0], self.eb(out_tensor, golden_out_tensor), "")


class DualGoldenCompare(DoubleCompare):
    """
    同时比较mare、mere、rmse的双标杆比较器
    """
    MARE = 10
    MERE = 2
    RMSE = 2

    def __init__(self, err: float, eb: float):
        super().__init__(err, eb)

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor, gpu_golden_out_tensor: torch.Tensor):
        result = {}
        npu_mare = self.mare(out_tensor, golden_out_tensor)
        gpu_mare = self.mare(out_tensor, golden_out_tensor)
        npu_mere = self.mere(out_tensor, golden_out_tensor)
        gpu_mere = self.mere(out_tensor, golden_out_tensor)
        npu_rmse = self.rmse(out_tensor, golden_out_tensor)
        gpu_rmse = self.rmse(out_tensor, golden_out_tensor)
        mare_result = npu_mare / max(gpu_mare, self.ERR)
        mere_result = npu_mere / max(gpu_mere, self.ERR)
        rmse_result = npu_rmse / max(gpu_rmse, self.ERR)
        result['mare'] = mare_result
        result['mere'] = mere_result
        result['rmse'] = rmse_result
        err = self.err(out_tensor, golden_out_tensor)
        eb = self.eb(out_tensor, golden_out_tensor)
        pass_ = all(mare_result <= self.MARE, mere_result <=
                    self.MERE, rmse_result <= self.RMSE)
        compare_result
        if not pass_:
            compare_result.pass_ = False
        return compare_result


class FloatHighPrecisionDualGoldenCompare(DoubleCompare):
    """
    高精度浮点计算的双标杆对比
    """

    def __call__(self, out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor, gpu_golden_out_tensor: torch.Tensor):
        result_npu = self.err(out_tensor, golden_out_tensor)
        result_gpu = self.err(out_tensor, gpu_golden_out_tensor)
        result = result_npu / \
                 torch.max(result_gpu, torch.full(result_npu.shape, self.ERR))
        pass_count = 0
        for err_value in result:
            if err_value < 2:
                pass_count += 1
        return CompareResult(pass_count / result.shape[0], self.eb(out_tensor, golden_out_tensor), "")


def get_compare(op_type: OpType, tensor_dtype: torch.dtype, compute_times: int = 1024,
                use_gpu: bool = False) -> BaseCompare:
    """根据算子类型和其它参数,  获取比较方法

    Args:
        op_type (OpTypes): 算子类型
        tensor_dtype (torch.dtype): Tensor数据类型
        compute_times (int, optional): 计算次数. Defaults to 1024.
        use_gpu (bool, optional): 是否使用GPU作为第二精度标准. Defaults to False.

    Returns:
        BaseCompare: _description_
    """
    if op_type in (OpType.COMPUTE_INTEGER, OpType.MOVE, OpType.CAST):
        # 搬运、整数计算、转换要求二进制一致
        return BinaryMatchCompare()
    if op_type == OpType.RAND:
        return RandomCompare()
    if op_type == OpType.COMPUTE_QUANT:
        return QuantCompare()
    if (op_type == OpType.CV_FUSION and use_gpu) or op_type == OpType.NA:
        return DualGoldenCompare()
    if op_type == OpType.COMPUTE_FLOAT_HIGH_PRECISION and use_gpu:
        return FloatHighPrecisionDualGoldenCompare()
    if op_type in (OpType.COMPUTE_FLOAT, OpType.COMPUTE_FLOAT_HIGH_PRECISION, OpType.VECTOR_FUSION):
        err_exp = ERR_EXP[op_type, tensor_dtype]
        eb_exp = EB_EXP[op_type, tensor_dtype]
        # 计算次数大于2048时，放开一定范围
        if compute_times >= 2048:
            err_exp += 1
        # 计算次数大于16384且选用fp32时，再次放开一定范围
        if compute_times >= 16384 and tensor_dtype == torch.float32:
            err_exp += 1
        return EBCompare(2 ** err_exp, 2 ** eb_exp)
