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
#include "mki/base/operation_base.h"
#include <set>
#include <cctype>
#include <algorithm>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/log/log.h"
#include "mki/utils/file_system/file_system.h"
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
    UNUSED_VALUE(specificParam);
    MKI_LOG(WARN) << opName_ << " use default GetInputNum function, input num: 1";
    return 1;
}

int64_t OperationBase::GetOutputNum(const Any &specificParam) const
{
    UNUSED_VALUE(specificParam);
    MKI_LOG(WARN) << opName_ << " use default GetOutputNum function, output num: 1";
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
