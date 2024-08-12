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
#ifndef MKI_UTILS_MATH_TENSOR_UTILS_H
#define MKI_UTILS_MATH_TENSOR_UTILS_H

#include <type_traits>
#include "securec.h"
#include "mki/tensor.h"
#include "mki/utils/log/log.h"
#include "mki/utils/SVector/SVector.h"

namespace Mki {
namespace Utils {
template <typename T_SRC, typename T_DST = T_SRC, typename T_CONT = SVector<T_SRC>>
size_t GetConstTensorSize(const T_CONT &tensorData)
{
    size_t originSize = sizeof(T_DST) * tensorData.size();
    constexpr size_t alignSize = 64;
    return (originSize + alignSize - 1) / alignSize * alignSize + alignSize; // round up to 64 and add 64 bytes bonus
}

uint64_t GetTensorAlignedSize(uint64_t len);
}
} // namespace Mki

#endif
