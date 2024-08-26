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
#ifndef MKI_LOADER_OP_SCHEDULE_H
#define MKI_LOADER_OP_SCHEDULE_H
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "mki/run_info.h"
#include "mki/operation.h"

namespace OpSpace {
using namespace Mki;

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
    void AddOperationKernels(const Operation *op);
    void AddKernel(Kernel const *kernel);

private:
    std::unordered_map<std::string, Operation *> opMap_;
    KernelMap kernelMap_;
};
} // namespace OpSpace

#endif
