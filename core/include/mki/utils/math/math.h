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

#ifndef MKI_UTILS_MATH_H
#define MKI_UTILS_MATH_H

#include <cstdint>

#include "mki/utils/log/log.h"

namespace Mki {
namespace Utils {
template <typename T>
T CeilDiv(T dividend, T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}
size_t RoundUp(size_t size, size_t divisor = 32);
size_t RoundDown(size_t size, size_t divisor = 32);
void SetRandseeds(uint32_t &randseed);
}
}

#endif // MKI_UTILS_MATH_H
