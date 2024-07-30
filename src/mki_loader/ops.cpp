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
#include "mki_loader/ops.h"
#include "op_schedule.h"

namespace OpSpace {
Ops::Ops() : opSchedule_(std::make_unique<OpSchedule>()) {}

Ops::~Ops() {}

Ops &Ops::Instance()
{
    static Ops instance;
    return instance;
}

std::vector<Operation *> Ops::GetAllOperations() const
{
    return opSchedule_->GetAllOperations();
}

Operation *Ops::GetOperationByName(const std::string &opName) const
{
    return opSchedule_->GetOperationByName(opName);
}

Kernel *Ops::GetKernelInstance(const std::string &kernelName) const
{
    return opSchedule_->GetKernelInstance(kernelName);
}

} // namespace OpSpace
