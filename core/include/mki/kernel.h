/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef MKI_KERNEL_H
#define MKI_KERNEL_H

#include <string>
#include <vector>
#include <cstddef>
#include "mki/launch_param.h"
#include "mki/run_info.h"
#include "mki/kernel_info.h"
#include "mki/utils/status/status.h"

namespace Mki {
enum KernelType : int {
    KERNEL_TYPE_AI_CORE = 0,
    KERNEL_TYPE_AI_CPU,
    KERNEL_TYPE_AIV,
    KERNEL_TYPE_WRITE_BACK,
    KERNEL_TYPE_MIX_AIC,
    KERNEL_TYPE_MIX_AIV,
    KERNEL_TYPE_FFTS_PLUS,
    KERNEL_TYPE_DSA,
    KERNEL_TYPE_DVPP,
    KERNEL_TYPE_HCCL,
    KERNEL_TYPE_INVALID
};

class Kernel {
public:
    Kernel() = default;
    virtual ~Kernel() = default;
    Kernel(const Kernel &kernel) = delete;
    virtual Kernel *Clone() const = 0;
    virtual std::string GetName() const = 0;
    virtual KernelType GetType() const = 0;
    virtual KernelInfo &GetKernelInfo() const = 0;

    virtual void SetLaunchWithTiling(bool flag) = 0;
    virtual bool GetLaunchWithTiling() const = 0;

    virtual void Reset() = 0;
    virtual Status Init(const LaunchParam &launchParam) = 0;
    virtual Status Run(const LaunchParam &launchParam, RunInfo &runInfo) = 0;

    virtual bool CanSupport(const LaunchParam &launchParam) const = 0;

    virtual uint64_t GetTilingSize(const LaunchParam &launchParam) const = 0;
};
using NewKernelFunc = const Kernel*(*)();
} // namespace Mki
#endif
