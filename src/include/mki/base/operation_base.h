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
#ifndef MKI_BASE_OPERATION_BASE_H
#define MKI_BASE_OPERATION_BASE_H

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include "mki/launch_param.h"
#include "mki/operation.h"

namespace Mki {
class OperationBase : public Operation {
public:
    explicit OperationBase(const std::string &opName) noexcept;
    ~OperationBase() override;
    std::string GetName() const override;
    Status InferShape(LaunchParam &launchParam) const override;

    int64_t GetInputNum(const Any &specificParam) const override;
    int64_t GetOutputNum(const Any &specificParam) const override;
    bool IsConsistent(const LaunchParam &launchParam) const override;

    const KernelList &GetKernelList() const override;
    Kernel *GetKernelByName(const std::string &kernelName) const override;
    void AddKernel(const std::string &kernelName, Kernel const *kernel);

protected:
    virtual Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const = 0;

protected:
    std::string opName_;
    KernelList kernelList_;
    KernelMap kernelMap_;
};
} // namespace Mki

#endif
