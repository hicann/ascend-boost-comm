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
#include <gtest/gtest.h>
#include "mki_loader/op_register.h"

namespace Mki {
TEST(OpRegisterTest, Base)
{
    std::vector<std::string> opList;
    auto &operationCreators = OpSpace::OperationRegister::GetOperationCreators();
    auto &kernelCreators = OpSpace::KernelRegister::GetKernelCreators();
    (void)operationCreators;
    (void)kernelCreators;
}
} // namespace Mki
