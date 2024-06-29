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
#ifndef CORE_LOADER_LOADER_H
#define CORE_LOADER_LOADER_H
#include <unordered_map>
#include <memory>
#include <atomic>
#include "mki/utils/noncopyable/noncopyable.h"
#include "mki/operation.h"
#include "mki/kernel.h"
#include "mki/bin_handle.h"

using namespace Mki;
namespace OpSpace {
class Loader : public NonCopyable {
public:
    Loader();
    ~Loader();
    void GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const;
    void GetOpKernels(const std::string &opName, KernelMap &kernels) const;
    bool IsValid() const;
private:
    void Load();
    bool LoadKernelBinarys();
    void CreateOperations();
    void CreateKernels();
    std::atomic_bool loadSuccess_{false};

private:
    std::unordered_map<std::string, Operation *> opMap_;
    std::unordered_map<std::string, KernelMap> opKernelMap_;
    std::unordered_map<std::string, BinHandle> binHandles_;
};
} // namespace OpSpace
#endif
