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
#ifndef CORE_BASE_KERNEL_BASE_H
#define CORE_BASE_KERNEL_BASE_H
#include <string>
#include <vector>
#include <functional>
#include <map>
#include "mki/kernel.h"
#include "mki/kernel_info.h"
#include "mki/run_info.h"
#include "mki/utils/rt/rt.h"

namespace Mki {
using KernelHandle = void**;

class KernelBase : public Kernel {
using KernelSelfCreator = std::function<KernelBase*(void)>;
public:
    KernelBase() = delete;
    KernelBase(const std::string &opName, KernelHandle handle);
    ~KernelBase() override;
    KernelBase(const KernelBase &) = delete;
    KernelBase &operator= (const KernelBase &) = delete;

    std::string GetName() const override;
    KernelType GetType() const override;
    KernelInfo &GetKernelInfo() const override;

    void SetLaunchWithTiling(bool flag) override;
    bool GetLaunchWithTiling() const override;

    void Reset() override;
    Status Init(const LaunchParam &launchParam) override;
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override;

    bool CanSupport(const LaunchParam &launchParam) const override;

    uint64_t GetTilingSize(const LaunchParam &launchParam) const override;
    Kernel *Clone() const override;

protected:
    virtual Status InitImpl(const LaunchParam &launchParam);
    KernelHandle GetHandle() const;
    void Copy(const KernelBase &other);
    uint32_t launchBufferSize_ = 0;

private:
    uint64_t GetKernelParamNum(const LaunchParam &launchParam);

    void UpdateKernelType();
    Status UpdateHwsyncArgs(void **args, uint64_t argsNum);
    Status UpdateConstTensorArgs(void **args, uint64_t argsNum,
                                 RtHostInputInfoT *info, uint64_t infoNum);
    Status UpdateConstTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo);
    Status UpdateInOutWkspArgs(void **args, uint64_t argsNum, const LaunchParam &launchParam, const RunInfo &runInfo);
    Status UpdateTilingArgs(RtArgsExT &argsEx, uint64_t argsNum, RunInfo &runInfo) const;
    Status UpdateTilingArgs(void **args, uint64_t argsNum, RunInfo &runInfo) const;
    Status MemsetTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo);

    std::string kernelName_;
    KernelType kernelType_{KernelType::KERNEL_TYPE_INVALID};
    bool launchWithTiling_{true};
    KernelHandle handle_{nullptr};
    mutable KernelInfo kInfo_;
    bool initFlag_{false};
    KernelSelfCreator creator_{nullptr};
    friend void SetKernelSelfCreator(KernelBase &kernel, KernelSelfCreator func);
};
} // namespace Mki

#endif
