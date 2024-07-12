/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    return (tensor.desc.Numel() == 0 && tensor.data == nullptr && tensor.hostData == nullptr && tensor.dataSize == 0);
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
