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
#include "mki/base/operation_base.h"
#include <set>
#include <cctype>
#include <algorithm>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/base/kernel_base.h"

namespace Mki {

OperationBase::OperationBase(const std::string &opName) noexcept : opName_(opName) {}

OperationBase::~OperationBase() {}

std::string OperationBase::GetName() const { return opName_; }

Status OperationBase::InferShape(LaunchParam &launchParam) const
{
    MKI_CHECK(launchParam.GetInTensorCount() == static_cast<size_t>(GetInputNum(launchParam.GetParam())),
        "input num is invalid, actual: " << launchParam.GetInTensorCount() << "expect : " <<
        GetInputNum(launchParam.GetParam()),
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "input num is invalid"));
    MKI_CHECK(launchParam.GetOutTensorCount() == static_cast<size_t>(GetOutputNum(launchParam.GetParam())),
        "output num is invalid, actual: " << launchParam.GetOutTensorCount() << "expect : " <<
        GetOutputNum(launchParam.GetParam()),
        return Status::FailStatus(ERROR_INFERSHAPE_ERROR, "output num is invalid"));
    return InferShapeImpl(launchParam, launchParam.GetOutTensors());
}

int64_t OperationBase::GetInputNum(const Any &specificParam) const
{
    (void)specificParam;
    return 1;
}

int64_t OperationBase::GetOutputNum(const Any &specificParam) const
{
    (void)specificParam;
    return 1;
}

bool OperationBase::IsConsistent(const LaunchParam &launchParam) const
{
    MKI_CHECK(launchParam.GetInTensorCount() == static_cast<size_t>(GetInputNum(launchParam.GetParam())),
        "input num is invalid: " << launchParam.GetInTensorCount(), return false);
    MKI_CHECK(launchParam.GetOutTensorCount() == static_cast<size_t>(GetOutputNum(launchParam.GetParam())),
        "output num is invalid: " << launchParam.GetOutTensorCount(), return false);
    SVector<Tensor> outTensors(GetOutputNum(launchParam.GetParam()), {});
    Status status = InferShapeImpl(launchParam, outTensors);
    MKI_CHECK(status.Ok(), "Failed to process infer shape", return false);
    return TensorsEqual(launchParam.GetOutTensors(), outTensors);
}

const KernelList &OperationBase::GetKernelList() const
{
    return kernelList_;
}

Kernel *OperationBase::GetKernelByName(const std::string &kernelName) const
{
    auto it = kernelMap_.find(kernelName);
    if (it == kernelMap_.end()) {
        MKI_LOG(ERROR) << "Kernel " << kernelName << " is not found, maybe it's not supported in this soc";
        return nullptr;
    }
    return (it->second)->Clone();
}

void OperationBase::AddKernel(const std::string &kernelName, Kernel const *kernel)
{
    MKI_CHECK(kernel != nullptr, "kernel is nullptr", return);
    kernelList_.push_back(kernel);
    kernelMap_[kernelName] = kernel;
}
} // namespace Mki
