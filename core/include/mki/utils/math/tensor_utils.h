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
#ifndef MKI_UTILS_MATH_TENSOR_UTILS_H
#define MKI_UTILS_MATH_TENSOR_UTILS_H

#include <type_traits>

#include "securec.h"

#include "mki/tensor.h"
#include "mki/utils/log/log.h"
#include "mki/utils/svector/svector.h"

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
