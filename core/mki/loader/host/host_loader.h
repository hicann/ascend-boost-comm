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
#ifndef CORE_LOADER_HOST_LOADER_H
#define CORE_LOADER_HOST_LOADER_H
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>
#include "mki/utils/noncopyable/noncopyable.h"
#include "mki/kernel.h"
#include "mki/operation.h"
#include "mki/utils/dl/dl.h"

namespace Mki {
class HostLoader : public NonCopyable {
public:
    HostLoader();
    ~HostLoader();
    void GetOpKernels(const std::string &opName, KernelMap &kernels) const;
    void GetAllOperations(std::unordered_map<std::string, Operation *> &ops) const;
private:
    void CreateOperations();
    void CreateKernels();

private:
    std::unordered_map<std::string, Operation *> opMap_;
    std::unordered_map<std::string, KernelMap> opKernelMap_;
};
} // namespace Mki
#endif
