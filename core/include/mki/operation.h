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
#ifndef MKI_OPERATION_H
#define MKI_OPERATION_H

#include <string>
#include <vector>
#include <unordered_map>
#include "mki/launch_param.h"
#include "mki/kernel.h"
#include "mki/utils/any/any.h"
#include "mki/utils/status/status.h"

namespace Mki {
using KernelList = std::vector<Kernel const *>;
using KernelMap = std::unordered_map<std::string, Kernel const *>;
class Operation {
public:
    Operation() = default;
    virtual ~Operation() = default;
    virtual std::string GetName() const = 0;
    virtual Status InferShape(LaunchParam &launchParam) const = 0;

    virtual int64_t GetInputNum(const Any &specificParam) const = 0;
    virtual int64_t GetOutputNum(const Any &specificParam) const = 0;
    virtual bool IsConsistent(const LaunchParam &launchParam) const = 0;

    virtual const KernelList &GetKernelList() const = 0;
    virtual Kernel *GetBestKernel(const LaunchParam &launchParam) const = 0;
    virtual Kernel *GetKernelByName(const std::string &kernelName) const = 0;
};
} // namespace Mki

#endif
