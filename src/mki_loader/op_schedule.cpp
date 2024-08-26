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
#include "op_schedule.h"
#include <fstream>
#include "mki/utils/log/log.h"
#include "mki/base/kernel_base.h"
#include "mki/utils/singleton/singleton.h"
#include "loader.h"

namespace OpSpace {
OpSchedule::OpSchedule() { AddAllOperations(); }

OpSchedule::~OpSchedule() {}

std::vector<Operation *> OpSchedule::GetAllOperations() const
{
    std::vector<Operation *> ops;
    for (const auto &it : opMap_) {
        if (it.second != nullptr) {
            ops.push_back(it.second);
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

void OpSchedule::AddOperationKernels(const Operation *op)
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
} // namespace OpSpace
