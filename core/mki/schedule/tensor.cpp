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
#include "mki/tensor.h"
#include "mki/utils/log/log.h"

namespace Mki {
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
