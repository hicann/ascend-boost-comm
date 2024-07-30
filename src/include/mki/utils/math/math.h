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

#ifndef MKI_UTILS_MATH_MATH_H
#define MKI_UTILS_MATH_MATH_H

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

#endif // MKI_UTILS_MATH_MATH_H
