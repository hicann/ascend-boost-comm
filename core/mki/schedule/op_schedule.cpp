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
#include "mki/op_schedule.h"
#include <fstream>
#include <tuple>
#include <type_traits>
#include "mki/utils/log/log.h"
#include "mki/utils/rt/rt.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/base/kernel_base.h"
#include "mki/utils/singleton/singleton.h"
#include "mki/loader/loader.h"

namespace Mki {
OpSchedule::OpSchedule() { AddAllOperations(); }

OpSchedule::~OpSchedule() {}

std::vector<Operation *> OpSchedule::GetAllOperations() const
{
    std::vector<Operation *> ops;
    for (const auto &[opName, op] : opMap_) {
        (void)opName;
        if (op != nullptr) {
            ops.push_back(op);
        }
    }
    return ops;
}

Operation *OpSchedule::GetOperationByName(const std::string &opName) const
{
    auto it = opMap_.find(opName);
    return it == opMap_.end() ? nullptr : it->second;
}

Kernel *OpSchedule::GetKernelInstance(const std::string &kernelName) const
{
    auto it = kernelMap_.find(kernelName);
    return it == kernelMap_.end() ? nullptr : (it->second)->Clone();
}

void OpSchedule::AddAllOperations()
{
    std::unordered_map<std::string, Operation *> ops;
    auto &loader = GetSingleton<Loader>();
    if (!loader.IsValid()) {
        return;
    }
    loader.GetAllOperations(ops);
    MKI_LOG(INFO) << "operation count:" << ops.size();
    for (const auto &[opName, op] : ops) {
        MKI_LOG(DEBUG) << "add op name:" << opName;
        AddOperationByName(op);
        AddOperationKernels(op);
    }
}

void OpSchedule::AddOperationByName(Operation *op)
{
    std::string opName = op->GetName();
    auto it = opMap_.find(opName);
    if (it == opMap_.end()) {
        opMap_.insert(std::make_pair(opName, op));
    } else {
        MKI_LOG(WARN) << "opName:" << opName << " repeat";
    }
}

void OpSchedule::AddOperationKernels(Operation *op)
{
    KernelList kernelList = op->GetKernelList();
    for (const auto &kernel : kernelList) {
        if (kernel == nullptr) {
            continue;
        }
        AddKernel(kernel);
    }
}

void OpSchedule::AddKernel(Kernel const *kernel)
{
    std::string kernelName = kernel->GetName();
    auto it = kernelMap_.find(kernelName);
    if (it == kernelMap_.end()) {
        kernelMap_.insert(std::make_pair(kernelName, kernel));
    } else {
        MKI_LOG(WARN) << "kernelName:" << kernelName << " repeat";
    }
}
} // namespace Mki
