# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

import numpy
import torch
from mkipythontest.constant import OpType
from mkipythontest.utils.dtype import is_float_type, is_int_type

ERR_BASE = {
    (OpType.COMPUTE_QUANT, torch.float16): -8,
    (OpType.COMPUTE_QUANT, torch.bfloat16): -7,
    (OpType.COMPUTE_QUANT, torch.float32): -11,
    (OpType.COMPUTE_FLOAT, torch.float16): -8,
    (OpType.COMPUTE_FLOAT, torch.bfloat16): -7,
    (OpType.COMPUTE_FLOAT, torch.float32): -11,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.float16): -11,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.bfloat16): -8,
    (OpType.COMPUTE_FLOAT_HIGH_PRECISION, torch.float32): -14,
    (OpType.VECTOR_FUSION, torch.float16): -9,
    (OpType.VECTOR_FUSION, torch.bfloat16): -8,
    (OpType.VECTOR_FUSION, torch.float32): -11,
    (OpType.CV_FUSION, torch.float16): -11,
    (OpType.CV_FUSION, torch.bfloat16): -8,
    (OpType.CV_FUSION, torch.float32): -14,
}

EB_BASE = {
    torch.float16: -10,
    torch.bfloat16: -7,
    torch.float32: -14
}


class CompareResult:

    def __init__(self, pass_num: int, total_num: int):
        self.pass_num: int = pass_num
        self.total_num: float = total_num
        self.other_info: dict = {}

    def add_item(self, item_name: str, item_actual_value: float, item_expected_value: float,
                 pass_: bool = True) -> None:
        """Add compare item

        :param item_name: item name
        :param item_actual_value: actual value
        :param item_expected_value: expected value
        :param pass_: if the item is passed, defaults to True
        """
        self.other_info[item_name] = {
            'actual_value': f"{item_actual_value:.3e}",
            'expected_value': f"{item_expected_value:.3e}",
            'pass': pass_
        }

    def __bool__(self) -> bool:
        return all([self.total_num <= 0 or self.pass_num == self.total_num]+list(map(lambda i: i['pass'], self.other_info.values())))

    def __str__(self) -> str:
        result = [f"Pass {self.pass_num}/{self.total_num}"]
        for item_name, item_info in self.other_info.items():
            result.append(
                f"{item_name}:{'PASS' if item_info['pass'] else 'FAIL'},{item_info['actual_value']}/{item_info['expected_value']}")
        return ', '.join(result)

    @property
    def pass_ratio(self) -> float:
        if self.total_num <= 0:
            return 1
        return self.pass_num / self.total_num


def ae(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> torch.Tensor:
    """Absolute Error
    """
    return torch.abs(torch.subtract(out_tensor, golden_out_tensor))


def re(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> torch.Tensor:
    """Relative Error
    """
    return torch.div(ae(out_tensor, golden_out_tensor), torch.abs(golden_out_tensor))


def mare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """MAx Relative Error
    """
    return torch.max(re(out_tensor, golden_out_tensor)).item()


def mere(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """MEan Relative Error
    """
    return torch.mean(re(out_tensor, golden_out_tensor)).item()


def rmse(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """Root Mean Square Error
    """
    abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
    return numpy.sqrt(torch.sum(torch.mul(abs_diff, abs_diff) / torch.numel(abs_diff)).numpy())


def eb(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """EB
    """
    out_tensor = out_tensor.float()
    golden_out_tensor = golden_out_tensor.float()
    diff = torch.sub(out_tensor, golden_out_tensor)
    golden_nmax = torch.clamp(torch.abs(golden_out_tensor), min=1)
    return torch.mean(torch.div(diff, golden_nmax)).item()


def err(out_tensor: torch.Tensor,
        golden_out_tensor: torch.Tensor,
        epsi: float = 1e-7) -> torch.Tensor:
    """Err for per values

    :param out_tensor: output tensor
    :param golden_out_tensor: golden output tensor
    :param epsi: a small value to avoid being divided by zero
    :return: a tensor of errors
    """
    out_tensor = out_tensor.float()
    golden_out_tensor = golden_out_tensor.float()
    # golden取绝对值并去0
    golden_out_tensor_abs_epsi = torch.clamp(
        torch.abs(golden_out_tensor), epsi)
    # 查找使用相对误差的index tensor
    use_relative_err_tensor = torch.gt(
        golden_out_tensor_abs_epsi, torch.ones(out_tensor.shape))
    # 计算各点绝对误差
    absolute_err_tensor = ae(golden_out_tensor, out_tensor)
    # 计算各点相对误差
    relative_err_tensor = torch.div(
        absolute_err_tensor, golden_out_tensor_abs_epsi)
    # 根据index tensor 将相对误差替换上去
    err_tensor = torch.where(use_relative_err_tensor ==
                             1, relative_err_tensor, absolute_err_tensor)
    # 原文档中 是除了再乘 可能有精度损失
    return err_tensor


class BaseComparer:
    def __init__(self):
        pass


class SingleComparer(BaseComparer):
    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor) -> CompareResult:
        pass


class DoubleComparer(BaseComparer):
    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor,
                golden_out_tensor_gpu: torch.Tensor) -> CompareResult:
        pass


class BinaryMatchComparer(SingleComparer):
    """Binary match compare for MOVE, CAST and COMPUTE_INTEGER
    """

    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor) -> CompareResult:
        pass_count = torch.sum(torch.eq(out_tensor, golden_out_tensor)).item()
        return CompareResult(pass_count, golden_out_tensor.numel())


class QuantIntComparer(SingleComparer):
    def __init__(self, max_ae: int = 1):
        self.max_ae = max_ae

    def compare(self, out_tensor, golden_out_tensor):
        ae_tensor = ae(out_tensor, golden_out_tensor)
        pass_count = torch.sum(
            torch.le(ae_tensor, torch.full(ae_tensor.shape, self.max_ae))).item()
        return CompareResult(pass_count, golden_out_tensor.numel())


class ErrEBComparer(SingleComparer):
    def __init__(self, err_value: float, eb_value: float):
        self.err_value = err_value
        self.eb_value = eb_value

    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor) -> CompareResult:
        err_result = err(out_tensor, golden_out_tensor)
        pass_count = torch.lt(err_result, torch.full(
            out_tensor.shape, self.err_value)).to(torch.int32).sum().item()
        result = CompareResult(pass_count, golden_out_tensor.numel())
        eb_result = eb(out_tensor, golden_out_tensor)
        result.add_item('eb', eb_result, self.eb_value,
                        eb_result < self.eb_value)
        return result


class DualGoldenComparer(DoubleComparer):
    def __init__(self,
                 err_value: float,
                 eb_value: float,
                 mare_value: float = 10,
                 mere_value: float = 2,
                 rmse_value: float = 2):
        self.err_value = err_value
        self.eb_value = eb_value
        self.mare_value = mare_value
        self.mere_value = mere_value
        self.rmse_value = rmse_value

    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor,
                golden_out_tensor_gpu: torch.Tensor) -> CompareResult:

        npu_mare = mare(out_tensor, golden_out_tensor)
        gpu_mare = mare(out_tensor, golden_out_tensor_gpu)
        npu_mere = mere(out_tensor, golden_out_tensor)
        gpu_mere = mere(out_tensor, golden_out_tensor_gpu)
        npu_rmse = rmse(out_tensor, golden_out_tensor)
        gpu_rmse = rmse(out_tensor, golden_out_tensor_gpu)
        mare_result = npu_mare / max(gpu_mare, self.err_value)
        mere_result = npu_mere / max(gpu_mere, self.err_value)
        rmse_result = npu_rmse / max(gpu_rmse, self.err_value)
        eb_result = eb(out_tensor, golden_out_tensor)
        eb_pass = eb_result < self.eb_value
        result = CompareResult(0, 0)
        result.add_item('mare', mare_result, self.mare_value,
                        mare_result < self.mare_value)
        result.add_item('mere', mere_result, self.mere_value,
                        mere_result < self.mere_value)
        result.add_item('rmse', rmse_result, self.rmse_value,
                        rmse_result < self.rmse_value)
        result.add_item('eb', eb_pass, self.eb_value, eb_pass)
        return result


class ComputeFloatHighPrecisionComparer(DoubleComparer):
    def __init__(self, err_value: float, eb_value: float):
        self.err_value = err_value
        self.eb_value = eb_value

    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor,
                golden_out_tensor_gpu: torch.Tensor):
        """Double golden compare for COMPUTE_FLOAT_HIGH_PRECISION.

        :param out_tensor: output tensor
        :param golden_out_tensor: golden output tensor
        :param golden_out_tensor_gpu: golden output tensor from GPU
        :param err_value: err standard
        :param eb_value: eb standard
        :return: compare result
        """
        err_result_cpu = err(out_tensor, golden_out_tensor)
        err_result_gpu = err(out_tensor, golden_out_tensor_gpu)
        err_result = torch.div(err_result_cpu, torch.max(
            err_result_gpu, torch.full(out_tensor.shape, self.err_value)))
        pass_count = torch.sum(torch.lt(err_result, torch.full(
            out_tensor.shape, 2)).to(torch.int32)).item()
        result = CompareResult(pass_count, out_tensor.numel())
        eb_result = eb(out_tensor, golden_out_tensor)
        result.add_item('eb', eb_result, self.eb_value,
                        eb_result < self.eb_value)
        return result


class DefaultSingleComparer(SingleComparer):
    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor) -> CompareResult:
        if is_int_type(out_tensor.dtype):
            pass_count = torch.eq(out_tensor, golden_out_tensor).sum().item()
            return CompareResult(pass_count, out_tensor.numel())
        if is_float_type(out_tensor.dtype):
            return ErrEBComparer(2**(-11), 2**(-14))(out_tensor, golden_out_tensor)


class DefaultDoubleComparer(DoubleComparer):
    def compare(self,
                out_tensor: torch.Tensor,
                golden_out_tensor: torch.Tensor,
                golden_out_tensor_gpu: torch.Tensor):
        return DefaultSingleComparer()(out_tensor, golden_out_tensor)



class ComparerFactory:
    def __init__(self):
        self.op_type: OpType = OpType.NA
        self.calculate_times: int = 1

    def set_op_type(self, op_type: OpType):
        self.op_type = op_type

    def set_calculate_times(self, calculate_times: int):
        self.calculate_times = calculate_times

    def get_single_golden_comparer(self, tensor_dtype: torch.dtype) -> SingleComparer:
        if self.op_type in (OpType.COMPUTE_INTEGER, OpType.MOVE, OpType.CAST):
            # 搬运、整数计算、转换要求二进制一致
            return BinaryMatchComparer()
        if self.op_type == OpType.COMPUTE_QUANT and is_int_type(tensor_dtype):
            # 量化为整数
            return QuantIntComparer()
        if (self.op_type in (OpType.COMPUTE_FLOAT, OpType.COMPUTE_FLOAT_HIGH_PRECISION, OpType.VECTOR_FUSION)
                or (self.op_type == OpType.COMPUTE_QUANT and is_float_type(tensor_dtype))):
            # 浮点计算、V融合、反量化为浮点
            err_exp = ERR_BASE[self.op_type, tensor_dtype]
            eb_exp = EB_BASE[tensor_dtype]
            # 计算次数大于2048时，放开一定范围
            if self.calculate_times >= 2048:
                err_exp += 1
            # 计算次数大于16384且选用fp32时，再次放开一定范围
            if self.calculate_times >= 16384 and tensor_dtype == torch.float32:
                err_exp += 1
            return ErrEBComparer(2 ** err_exp, 2 ** eb_exp)
        logging.info(f"Comparer for {self.op_type.name} in {tensor_dtype} without GPU golden result is not found. Use default comparer.")
        return DefaultSingleComparer()

    def get_double_golden_comparer(self, tensor_dtype: torch.dtype) -> DoubleComparer:
        if self.op_type == OpType.COMPUTE_FLOAT_HIGH_PRECISION:
            # 高精度
            return ComputeFloatHighPrecisionComparer(2 ** ERR_BASE[self.op_type, tensor_dtype], 2 ** EB_BASE[tensor_dtype])
        if self.op_type in (OpType.COMPUTE_FLOAT, OpType.CV_FUSION):
            # 浮点、CV
            return DualGoldenComparer(2 ** ERR_BASE[self.op_type, tensor_dtype], 2 ** EB_BASE[tensor_dtype])
        logging.info(f"Comparer for {self.op_type.name} in {tensor_dtype} with GPU golden result is not found. Use default comparer.")
        return DefaultDoubleComparer()
