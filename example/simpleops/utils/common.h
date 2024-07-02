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
#ifndef ASCEND_OPS_COMMON_H
#define ASCEND_OPS_COMMON_H

#include "asdops/utils/svector/svector.h"

namespace AsdOps {
template<class T>
bool CheckParamAllPositive(SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] <= 0) {
            return false;
        }
    }
    return true;
}
template<class T>
bool CheckParamAnyNegative(SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] < 0) {
            return true;
        }
    }
    return false;
}
} // namespace AsdOps

#endif