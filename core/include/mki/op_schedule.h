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
#ifndef CORE_SCHEDULE_OPSCHEDULE_H
#define CORE_SCHEDULE_OPSCHEDULE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "mki/run_info.h"
#include "mki/operation.h"

namespace Mki {

class OpSchedule {
public:
    OpSchedule();
    ~OpSchedule();
    std::vector<Operation *> GetAllOperations() const;
    Operation *GetOperationByName(const std::string &opName) const;
    Kernel *GetKernelInstance(const std::string &kernelName) const;

private:
    void AddAllOperations();
    void AddOperationByName(Operation *op);
    void AddOperationKernels(Operation *op);
    void AddKernel(Kernel const *kernel);

private:
    std::unordered_map<std::string, Operation *> opMap_;
    KernelMap kernelMap_;
};
} // namespace Mki

#endif
