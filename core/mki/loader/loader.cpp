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
#include "loader.h"
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"
#include "mki/loader/host/host_loader.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

namespace Mki {
Loader::Loader() : opLoader_(std::make_unique<HostLoader>()) { Load(); }

Loader::~Loader() {}

void Loader::GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const
{
    ops = ops_;
}

bool Loader::IsValid() const
{
    return loadSuccess_;
}

void Loader::Load()
{
    loadSuccess_ = false;
    opLoader_->GetAllOperations(ops_);

    for (const auto &[opName, op] : ops_) {
        MKI_LOG(DEBUG) << "mki load operation: " << opName;
        OperationBase *opBase = reinterpret_cast<OperationBase *>(op);
        MKI_CHECK(opBase != nullptr, opName << ": opBase is nullptr", return);
        KernelMap nameKernelsMap;
        opLoader_->GetOpKernels(op->GetName(), nameKernelsMap);
        for (const auto &[kernelName, kernel] : nameKernelsMap) {
            opBase->AddKernel(kernelName, kernel);
        }
    }
    loadSuccess_ = true;
}
} // namespace Mki
