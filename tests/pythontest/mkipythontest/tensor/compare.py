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
from functools import partial

import numpy
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
    (OpType.CV_FUSION, torch.float16): -11,
    (OpType.CV_FUSION, torch.bfloat16): -8,
    (OpType.CV_FUSION, torch.float32): -14,
}

EB_EXP = {
    torch.float16: -10,
    torch.bfloat16: -7,
    torch.float32: -14
}


class CompareResult:
    """
    比较结果
    """

    def __init__(self, pass_num: int, total_num: int):
        self.pass_num: int = pass_num
        self.total_num: float = total_num
        self.other_info: dict = {}

    def add_item(self, item_name: str, item_actual_value: float, item_expected_value: float, pass_: bool = True) -> None:
        self.other_info[item_name] = {
            'actual_value': item_actual_value,
            'expected_value': item_expected_value,
            'pass': pass_
        }

    def __bool__(self) -> bool:
        return self.pass_num == self.total_num

    @property
    def pass_ratio(self) -> float:
        return self.pass_num / self.total_num

    @property
    def status(self) -> str:
        if self:
            if all(map(lambda i: i['pass'], self.other_info.values())):
                return "OK"
            else:
                return "WARNING"
        return "FAILED"


def results_to_dataframe(results: list[CompareResult]) -> dict[str, str]:
    f_pass_all = "YES" if all(results) else "NO"
    f_pass = ';'.join(map(lambda r: r.status, results))
    f_pass_ratio = ';'.join(map(lambda r: "%.3f" % r.pass_ratio, results))
    f_other_info = ';'.join(map(lambda r: r.other_info, results))
    return {
        'AllPass': f_pass_all,
        'Pass': f_pass,
        'PassRatio': f_pass_ratio,
        'OtherInfo': f_other_info}


def mare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """
    计算最大误差

    :param out_tensor:
    :param golden_out_tensor:
    :return:
    """
    abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
    return torch.max(torch.div(abs_diff, torch.abs(golden_out_tensor))).item()


def mere(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """
    计算平均误差

    :param out_tensor:
    :param golden_out_tensor:
    :return:
    """
    abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
    return torch.mean(torch.div(abs_diff, torch.abs(golden_out_tensor))).item()


def rmse(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """
    计算均方根误差

    :param out_tensor: 输出张量
    :param golden_out_tensor: 标杆张量
    :return:
    """
    abs_diff = torch.abs(torch.subtract(out_tensor, golden_out_tensor))
    return numpy.sqrt(torch.sum(torch.mul(abs_diff, abs_diff) / torch.numel(abs_diff)).numpy())


def eb(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> float:
    """
    计算误差变异率

    :param out_tensor:
    :param golden_out_tensor:
    :return:
    """
    diff = torch.sub(out_tensor, golden_out_tensor)
    golden_nmax = torch.clamp(torch.abs(golden_out_tensor), min=1)
    return torch.mean(diff, golden_nmax).item()


def err(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor, epsi: float = 1e-7) -> torch.Tensor:
    """

    :param epsi:
    :param err:
    :param out_tensor:
    :param golden_out_tensor:
    :return:
    """
    # golden取绝对值并去0
    golden_out_tensor_abs_epsi = torch.where(
        golden_out_tensor == 0, epsi, golden_out_tensor)
    # 查找使用相对误差的index tensor
    use_relative_err_tensor = torch.gt(
        golden_out_tensor, torch.full(out_tensor.shape, 1.0))
    # 计算各点绝对误差
    absolute_err_tensor = torch.abs(torch.sub(golden_out_tensor, out_tensor))
    # 计算各点相对误差
    relative_err_tensor = torch.div(
        absolute_err_tensor, golden_out_tensor_abs_epsi)
    # 根据index tensor 将相对误差替换上去
    err_tensor = torch.where(use_relative_err_tensor ==
                             1, relative_err_tensor, absolute_err_tensor)
    # 原文档中 是除了再乘 可能有精度损失
    return err_tensor


def binary_match_compare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor) -> CompareResult:
    pass_count = torch.sum(torch.eq(out_tensor, golden_out_tensor)).item()
    result = CompareResult(pass_count, golden_out_tensor.numel())
    return result


def err_eb_compare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor, err_value: float,
                   eb_value: float) -> CompareResult:
    err_result = err(out_tensor, golden_out_tensor)
    pass_count = torch.lt(err_result, torch.full(
        out_tensor.shape, err_value)).to(torch.int32).sum().item()
    result = CompareResult(pass_count, golden_out_tensor.numel())
    eb_result = eb(out_tensor, golden_out_tensor)
    result.add_item('eb', eb_result, eb_value, eb_result < eb_value)
    return result


def dual_compare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor, golden_out_tensor_gpu: torch.Tensor,
                 err_value: float, eb_value: float, mare_value: float = 10, mere_value: float = 2,
                 rmse_value: float = 2) -> CompareResult:
    result = {}
    npu_mare = mare(out_tensor, golden_out_tensor)
    gpu_mare = mare(out_tensor, golden_out_tensor_gpu)
    npu_mere = mere(out_tensor, golden_out_tensor)
    gpu_mere = mere(out_tensor, golden_out_tensor_gpu)
    npu_rmse = rmse(out_tensor, golden_out_tensor)
    gpu_rmse = rmse(out_tensor, golden_out_tensor_gpu)
    mare_result = npu_mare / max(gpu_mare, err_value)
    mere_result = npu_mere / max(gpu_mere, err_value)
    rmse_result = npu_rmse / max(gpu_rmse, err_value)
    mare_pass = mare_result < mare_value
    mere_pass = mere_result < mere_value
    rmse_pass = rmse_result < rmse_value
    eb_result = eb(out_tensor, golden_out_tensor)
    eb_pass = eb_result < eb_value
    result = CompareResult(
        [mare_pass, mere_pass, rmse_pass, eb_pass].count(True), 4)
    result.add_item('mare', mare_result, mare_value, mare_pass)
    result.add_item('mere', mere_result, mere_value, mere_pass)
    result.add_item('rmse', rmse_result, rmse_value, rmse_pass)
    result.add_item('eb', eb_pass, eb_value, eb_pass)
    return result


def float_high_precision_compare(out_tensor: torch.Tensor, golden_out_tensor: torch.Tensor,
                                 gpu_golden_out_tensor: torch.Tensor, err_value: float, eb_value: float):
    err_result_cpu = err(out_tensor, golden_out_tensor)
    err_result_gpu = err(out_tensor, gpu_golden_out_tensor)
    err_result = torch.div(err_result_cpu, torch.max(
        err_result_gpu, torch.full(out_tensor.shape, err_value)))
    pass_count = torch.sum(torch.lt(err_result, torch.full(
        out_tensor.shape, 2)).to(torch.int32)).item()
    result = CompareResult(pass_count, out_tensor.numel())
    eb_result = eb(out_tensor, golden_out_tensor)
    result.add_item('eb', eb_result, eb_value, eb_result < eb_value)
    return result


def dummy_compare(op_type: OpType, tensor_dtype: torch.dtype, use_gpu_golden: bool = False, *args,
                  **kwargs) -> CompareResult:
    logging.info(
        f"Comparer for {op_type.name} in {tensor_dtype} with {['out', ''][int(use_gpu_golden)]} GPU golden result is not implemented. Returns PASS.")
    return CompareResult(1, 1)


class ComparerFactory:
    def __init__(self):
        self.op_type: OpType = OpType.NA
        self.calculate_times: int = 1

    def set_op_type(self, op_type: OpType):
        self.op_type = op_type

    def set_calculate_times(self, calculate_times: int):
        self.calculate_times = calculate_times

    def get_single_golden_comparer(self, tensor_dtype: torch.dtype) -> callable:
        if self.op_type in (OpType.COMPUTE_INTEGER, OpType.MOVE, OpType.CAST):
            # 搬运、整数计算、转换要求二进制一致
            return binary_match_compare
        if self.op_type == OpType.COMPUTE_QUANT:
            pass
        if self.op_type in (OpType.COMPUTE_FLOAT, OpType.COMPUTE_FLOAT_HIGH_PRECISION, OpType.VECTOR_FUSION):
            err_exp = ERR_EXP[self.op_type, tensor_dtype]
            eb_exp = EB_EXP[tensor_dtype]
            # 计算次数大于2048时，放开一定范围
            if self.calculate_times >= 2048:
                err_exp += 1
            # 计算次数大于16384且选用fp32时，再次放开一定范围
            if self.calculate_times >= 16384 and tensor_dtype == torch.float32:
                err_exp += 1
            return partial(err_eb_compare, {'err_value': 2 ** err_exp, 'eb_value': 2 ** eb_exp})
        return dummy_compare(self.op_type, tensor_dtype, use_gpu_golden=False)

    def get_double_golden_comparer(self, tensor_dtype: torch.dtype) -> callable:
        if self.op_type == OpType.COMPUTE_FLOAT_HIGH_PRECISION:
            return partial(float_high_precision_compare, {'err_value': 2 ** ERR_EXP[self.op_type, tensor_dtype],
                                                          'eb_value': 2 ** EB_EXP[tensor_dtype]})
        if self.op_type in (OpType.COMPUTE_FLOAT, OpType.CV_FUSION):
            return partial(dual_compare, {'err_value': 2 ** ERR_EXP[self.op_type, tensor_dtype],
                                          'eb_value': 2 ** EB_EXP[tensor_dtype]})
        return dummy_compare(self.op_type, tensor_dtype, use_gpu_golden=True)
