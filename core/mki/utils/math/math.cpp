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

#include "mki/utils/math/math.h"
#include <sys/time.h>
namespace Mki {
namespace Utils {
size_t RoundUp(size_t size, size_t divisor)
{
    if (divisor == 0) {
        return size;
    }
    return (size + divisor - 1) / divisor * divisor;
}

size_t RoundDown(size_t size, size_t divisor)
{
    if (divisor == 0) {
        return size;
    }
    return size / divisor * divisor;
}

void SetRandseeds(uint32_t &randseed)
{
    if (randseed == 0xffffffff) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        randseed = static_cast<uint32_t>(tv.tv_sec * 1000 + tv.tv_usec / 1000); // 1000 convert in us/ms/s
    }
    srand(randseed);
}

}
}
