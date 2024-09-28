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
#ifndef MKI_BASE_KERNEL_BASE_H
#define MKI_BASE_KERNEL_BASE_H
#include <string>
#include <functional>
#include <map>
#include "mki/kernel.h"
#include "mki/kernel_info.h"
#include "mki/run_info.h"
#include "mki/bin_handle.h"

namespace Mki {
class KernelBase : public Kernel {
using KernelSelfCreator = std::function<KernelBase*(void)>;
public:
    KernelBase() = delete;
    KernelBase(const KernelBase &) = delete;
    KernelBase &operator=(const KernelBase &) = delete;
    KernelBase(const std::string &opName, const BinHandle *handle);
    ~KernelBase() override;
    Kernel *Clone() const override;
    void Reset() override;

    bool CanSupport(const LaunchParam &launchParam) const override;
    uint64_t GetTilingSize(const LaunchParam &launchParam) const override;
    Status Init(const LaunchParam &launchParam) override;
    Status Run(const LaunchParam &launchParam, RunInfo &runInfo) override;

    void SetLaunchWithTiling(bool flag) override;
    void SetTilingHostAddr(uint8_t *addr, uint64_t len) override;
    std::string GetName() const override;
    const KernelInfo &GetKernelInfo() const override;
    KernelType GetType() const override;

protected:
    virtual Status InitImpl(const LaunchParam &launchParam);
    const BinHandle *GetBinHandle() const;
    void Copy(const KernelBase &other);

protected:
    uint32_t launchBufferSize_ = 0;
    KernelInfo kernelInfo_;

private:
    uint64_t GetKernelArgsNum(const LaunchParam &launchParam);

private:
    std::string kernelName_;
    KernelType kernelType_{KernelType::KERNEL_TYPE_INVALID};
    const BinHandle *handle_{nullptr};
    KernelSelfCreator creator_{nullptr};
    friend void SetKernelSelfCreator(KernelBase &kernel, KernelSelfCreator func);
};

void SetKernelSelfCreator(KernelBase &kernel, KernelBase::KernelSelfCreator func);
} // namespace Mki

#endif
