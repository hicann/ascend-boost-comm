/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MKI_LOADER_LOADER_H
#define MKI_LOADER_LOADER_H
#include <unordered_map>
#include <memory>
#include <atomic>
#include "mki/operation.h"
#include "mki/kernel.h"
#include "mki/bin_handle.h"

namespace OpSpace {
class Loader {
public:
    Loader();
    ~Loader();
    Loader(const Loader &) = delete;
    Loader &operator=(const Loader &other) = delete;
    void GetAllOperations(std::unordered_map<std::string, Mki::Operation *> &ops) const;
    void GetOpKernels(const std::string &opName, Mki::KernelMap &kernels) const;
    bool IsValid() const;

private:
    void Load();
    bool LoadKernelBinarys();
    bool CreateOperations();
    bool CreateKernels();
    bool OpBaseAddKernels() const;

private:
    std::atomic_bool loadSuccess_{false};
    std::unordered_map<std::string, Mki::Operation *> opMap_;
    std::unordered_map<std::string, Mki::KernelMap> opKernelMap_;
    std::unordered_map<std::string, Mki::BinHandle> binHandles_;
};
} // namespace OpSpace
#endif
