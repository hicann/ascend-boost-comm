/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/tensor.h"
#include "mki/utils/log/log.h"

namespace Mki {
int64_t CalNumel(const Mki::SVector<int64_t> &value, uint32_t start, uint32_t end)
{
    int64_t elementCount = 1;
    int64_t maxVal = std::numeric_limits<int64_t>::max();
    int64_t maxDimValue = std::numeric_limits<int32_t>::max();
    for (uint32_t idx = start; idx < end; idx++) {
        int64_t tmpVal = value[idx];
        if (tmpVal <= 0 || tmpVal > maxDimValue) {
            MKI_LOG(ERROR) << "dims : " << tmpVal << " is invalid!";
            return 0;
        }
        if (maxVal / elementCount < tmpVal) {
            MKI_LOG(ERROR) << "Tensor size is overflow!";
            return 0;
        }
        elementCount *= tmpVal;
    }
    return elementCount;
}

int64_t TensorDesc::Numel() const
{
    if (dims.size() == 0) {
        return 0;
    }
    if (dims.size() == 1 && dims[0] == 0) {
        return 0;
    }

    return CalNumel(dims, 0, dims.size());
}

void TensorDesc::View(const Mki::SVector<int64_t> &newDims)
{
    int64_t elementCount = CalNumel(newDims, 0, newDims.size());
    if (elementCount == 0) {
        return;
    }

    int64_t oldElementCount = Numel();
    if (elementCount != oldElementCount) {
        MKI_LOG(ERROR) << "tesnor view fail, elementCount:" << elementCount << " not equal:" << oldElementCount;
        return;
    }
    dims = newDims;
}

std::string TensorDesc::ToString() const
{
    std::stringstream ss;
    ss << "dtype:" << GetStrWithDType(dtype) << ", format:" << GetStrWithFormat(format) << ", dims:[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i == 0) {
            ss << dims.at(i);
        } else {
            ss << ", " << dims.at(i);
        }
    }
    ss << "]";

    return ss.str();
}

int64_t Tensor::Numel() const { return desc.Numel(); }

void Tensor::View(const Mki::SVector<int64_t> &newDims) { return desc.View(newDims); }

std::string Tensor::ToString() const
{
    std::stringstream ss;
    ss << desc.ToString() << ", dataSize:" << dataSize;
#ifdef _DEBUG
    ss << ", data:" << data;
#else
    ss << ", data:" << (data != nullptr);
#endif
    return ss.str();
}
} // namespace Mki
