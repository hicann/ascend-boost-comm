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
#ifndef CORE_BASE_OPERATIONBASE_H
#define CORE_BASE_OPERATIONBASE_H

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
    void AddKernel(const std::string &kernelName, Kernel const *kernel); // TODO: protected

protected:
    virtual Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const = 0;
    friend class Loader;

protected:
    std::string opName_;
    KernelList kernelList_;
    KernelMap kernelMap_;
};
} // namespace Mki

#endif
