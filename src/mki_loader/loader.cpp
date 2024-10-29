/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "loader.h"
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"
#include "mki/bin_handle.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki_loader/op_register.h"
#include "mki/utils/platform/platform_info.h"

namespace OpSpace {
Loader::Loader() { Load(); }

Loader::~Loader() {}

void Loader::GetAllOperations(std::unordered_map<std::string, Mki::Operation *> &ops) const
{
    ops = opMap_;
}

void Loader::GetOpKernels(const std::string &opName, Mki::KernelMap &kernels) const
{
    auto it = opKernelMap_.find(opName);
    if (it != opKernelMap_.end()) {
        kernels = it->second;
    }
}

bool Loader::IsValid() const
{
    return loadSuccess_;
}

bool Loader::CreateOperations()
{
    auto &operationCreators = OperationRegister::GetOperationCreators();
    for (const auto &opCreator : operationCreators) {
        Mki::Operation *operation = opCreator();
        MKI_CHECK(operation != nullptr, "create operation fail", return false);
        opMap_[operation->GetName()] = operation;
        MKI_LOG(DEBUG) << "Create operation " << operation->GetName();
    }
    return true;
}

bool Loader::CreateKernels()
{
    auto &kernelCreators = KernelRegister::GetKernelCreators();
    for (const auto &creatorInfo : kernelCreators) {
        const auto &kernelName = creatorInfo.kernelName;
        auto it = binHandles_.find(kernelName);
        if (it == binHandles_.end()) {
            MKI_LOG(WARN) << kernelName << " find bin handle fail";
            continue;
        }
        auto &handle = it->second;
        MKI_CHECK(handle.Init(kernelName), kernelName << " init handle fail", continue);

        auto kernelCreator = creatorInfo.func;
        MKI_CHECK(kernelCreator, kernelName << " creator function is null", continue);
        const Mki::Kernel *kernel = kernelCreator(&handle);
        MKI_CHECK(kernel != nullptr, "Invalid kernel found in op: " << kernelName, return false);

        const auto &opName = creatorInfo.opName;
        auto &opKernel = opKernelMap_[opName];
        opKernel[kernelName] = kernel;
    }
    return true;
}

bool Loader::LoadKernelBinarys()
{
    std::string deviceVersion = Mki::PlatformInfo::Instance().GetPlatformName();
    MKI_CHECK(deviceVersion != "unrecognized", "Get device soc version fail: " << deviceVersion, return false);

    const auto &kernelBinaryMap = KernelBinaryRegister::GetKernelBinaryMap();
    for (auto &item : kernelBinaryMap) {
        auto &binaryList = item.second;
        for (auto &binary : binaryList) {
            if (binary.targetSoc == deviceVersion) {
                binHandles_.emplace(item.first, &binary);
                MKI_LOG(DEBUG) << "Kernel " << deviceVersion << " find basic information success";
                break;
            }
        }
    }
    MKI_LOG(DEBUG) << "Loaded kernel Count: " << binHandles_.size();
    return true;
}

bool Loader::OpBaseAddKernels() const
{
    for (const auto &[opName, op] : opMap_) {
        MKI_LOG(DEBUG) << "mki load operation: " << opName;
        Mki::OperationBase *opBase = dynamic_cast<Mki::OperationBase *>(op);
        MKI_CHECK(opBase != nullptr, opName << ": opBase is nullptr", return false);
        auto it = opKernelMap_.find(opName);
        if (it == opKernelMap_.end()) {
            MKI_LOG(WARN) << opName << ": find kernels map fail ";
            continue;
        }
        auto &nameKernelsMap = it->second;
        for (const auto &[kernelName, kernel] : nameKernelsMap) {
            opBase->AddKernel(kernelName, kernel);
        }
    }
    return true;
}

void Loader::Load()
{
    loadSuccess_ = false;
    MKI_LOG(INFO) << "register flag: " << g_opsRegisterFlag << " is set";

    MKI_CHECK(LoadKernelBinarys(), "Load kernel binarys fail", return);
    MKI_CHECK(CreateOperations(), "Load operations fail", return);
    MKI_CHECK(CreateKernels(), "Load kernels fail", return);
    MKI_CHECK(OpBaseAddKernels(), "OpBase add kernels fail", return);

    loadSuccess_ = true;
}
} // namespace OpSpace
