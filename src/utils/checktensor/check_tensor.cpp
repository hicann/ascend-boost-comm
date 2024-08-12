/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * MindKernelInfra is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "mki/utils/checktensor/check_tensor.h"
#include <limits>
#include "mki/utils/assert/assert.h"

namespace Mki {
constexpr uint32_t MIN_DIM = 1;
constexpr uint32_t MAX_DIM = 8;
constexpr int64_t MAX_SHAPE_SIZE = 256uLL * 1024uLL * 1024uLL * 1024uLL; // 256GB

bool CheckInTensor(const TensorDesc &tensorDesc)
{
    size_t tensorSize = 1;
    uint64_t dimsSize = tensorDesc.dims.size();
    MKI_CHECK(dimsSize >= MIN_DIM && dimsSize <= MAX_DIM, "dims Size is invalid: " << dimsSize, return false);

    size_t elementCount = static_cast<size_t>(tensorDesc.Numel());
    MKI_CHECK(elementCount != 0, "Get Tensor Numel failed", return false);

    size_t elementSize = GetTensorElementSize(tensorDesc.dtype);
    MKI_CHECK(elementSize != 0, "Get Tensor ElementSize failed, dtype : " << tensorDesc.dtype, return false);
    size_t maxVal = std::numeric_limits<uint64_t>::max();
    tensorSize = elementSize * elementCount;
    MKI_CHECK(elementCount <= maxVal / elementSize && tensorSize <= MAX_SHAPE_SIZE,
        "tensorSize is invalid : " << tensorSize, return false);

    return true;
}

bool CheckInTensors(const LaunchParam &launchParam)
{
    bool ret = true;
    for (size_t i = 0; i < launchParam.GetInTensorCount(); ++i) {
        const Tensor &tensor = launchParam.GetInTensor(i);
        ret = CheckEmptyTensor(tensor);
        if (ret) {
            continue;
        }
        ret = CheckInTensor(tensor.desc);
        if (!ret) {
            return ret;
        }
    }

    return ret;
}

bool CheckEmptyTensor(const Tensor &tensor)
{
    return (tensor.desc.Numel() == 0 && tensor.data == nullptr &&
            tensor.hostData == nullptr && tensor.dataSize == 0);
}

bool TensorsEqual(const SVector<Tensor> &lhs, const SVector<Tensor> &rhs)
{
    if (lhs.size() != rhs.size()) {
        MKI_LOG(ERROR) << "Tensor size " << lhs.size() << " and " << rhs.size() << " not equal";
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].desc.dtype != rhs[i].desc.dtype) {
            MKI_LOG(ERROR) << "Failed to compare dtype of tensor" << i << ", dtype0 : "
                << GetStrWithDType(lhs[i].desc.dtype) << ", dtype1 : "
                << GetStrWithDType(rhs[i].desc.dtype);
            return false;
        }
        if (lhs[i].desc.format != rhs[i].desc.format) {
            MKI_LOG(ERROR) << "Failed to compare format of tensor" << i << ", format0 : "
                << GetStrWithFormat(lhs[i].desc.format) << ", format1 : "
                << GetStrWithFormat(rhs[i].desc.format);
            return false;
        }
        if (lhs[i].desc.dims != rhs[i].desc.dims) {
            MKI_LOG(ERROR) << "Failed to compare dims of tensor" << i << ", dims0 : " << lhs[i].desc.dims <<
                ", dims1 : " << rhs[i].desc.dims;
            return false;
        }
    }

    return true;
}
} // namespace Mki
