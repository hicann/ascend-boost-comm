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
#ifndef MKI_KERNEL_H
#define MKI_KERNEL_H

#include <string>
#include "mki/launch_param.h"
#include "mki/run_info.h"
#include "mki/kernel_info.h"
#include "mki/utils/status/status.h"

namespace Mki {
enum KernelType : int {
    KERNEL_TYPE_INVALID = -1,
    KERNEL_TYPE_AI_CORE,
    KERNEL_TYPE_AI_CPU,
    KERNEL_TYPE_AIV,
    KERNEL_TYPE_WRITE_BACK,
    KERNEL_TYPE_MIX_AIC,
    KERNEL_TYPE_MIX_AIV,
    KERNEL_TYPE_FFTS_PLUS,
    KERNEL_TYPE_DSA,
    KERNEL_TYPE_DVPP,
    KERNEL_TYPE_HCCL,
};

class Kernel {
public:
    Kernel() = default;
    Kernel(const Kernel &kernel) = delete;
    virtual ~Kernel() = default;
    virtual Kernel *Clone() const = 0;
    virtual void Reset() = 0;

    virtual bool CanSupport(const LaunchParam &launchParam) const = 0;
    virtual uint64_t GetTilingSize(const LaunchParam &launchParam) const = 0;
    virtual Status Init(const LaunchParam &launchParam) = 0;
    virtual Status Run(const LaunchParam &launchParam, RunInfo &runInfo) = 0;

    virtual void SetLaunchWithTiling(bool flag) = 0;
    virtual void SetTilingHostAddr(uint8_t *addr, uint64_t len) = 0;
    virtual std::string GetName() const = 0;
    virtual const KernelInfo &GetKernelInfo() const = 0;
    virtual KernelType GetType() const = 0;
};
} // namespace Mki
#endif
