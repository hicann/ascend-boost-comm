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
#include "host_loader.h"
#include <cstring>
#include "mki/kernel.h"
#include "mki/base/kernel_base.h"
#include "mki/base/op_register.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/strings/str_checker.h"
#include "mki/utils/singleton/singleton.h"
#include "mki/loader/device_kernel/device_kernel_loader.h"

namespace Mki {

HostLoader::HostLoader()
{
    CreateOperations();
    CreateKernels();
    MKI_LOG(DEBUG) << "Operation & kernel loaded!";
}

HostLoader::~HostLoader() {}

void HostLoader::GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const
{
    ops = opMap_;
}

void HostLoader::GetOpKernels(const std::string &opName, KernelMap &kernels) const
{
    auto it = opKernelMap_.find(opName);
    if (it != opKernelMap_.end()) {
        kernels = it->second;
    }
}

void HostLoader::CreateOperations()
{
    auto &operationCreators = OperationRegister::OperationCreators();
    for (const auto &opCreator : operationCreators) {
        Operation *operation = opCreator();
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
}

void HostLoader::CreateKernels()
{
    auto &kernelCreators = KernelRegister::KernelCreators();
    for (const auto &[kernelCreator, opName] : kernelCreators) {
        auto &opKernel = opKernelMap_[opName];
        const Kernel *kernel = kernelCreator();
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << opName, continue);
        if (GetSingleton<KernelBinaryLoader>().HasKernelMetaInfo(kernel->GetName())) {
            opKernel[kernel->GetName()] = kernel;
        }
    }
}
} // namespace Mki
