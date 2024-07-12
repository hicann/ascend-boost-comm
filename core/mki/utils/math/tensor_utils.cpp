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
#include "mki/utils/math/tensor_utils.h"

#include "mki/utils/math/math.h"

namespace Mki {
namespace Utils {
uint64_t GetTensorAlignedSize(uint64_t len)
{
    constexpr uint64_t alignTo = 64;
    if (len == 0) {
        return 0;
    }
    return RoundUp(len, alignTo);
}
}
}
