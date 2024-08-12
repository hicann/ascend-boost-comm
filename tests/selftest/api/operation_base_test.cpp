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
#include "mki/base/operation_base.h"
#include "mki/base/kernel_base.h"

namespace {
using namespace Mki;

class OperationNew : public OperationBase {
public:
    OperationNew() : OperationBase("") {}
    Mki::Status InferShape(LaunchParam &launchParam) const override { return Status::OkStatus(); };
    int64_t GetInputNum(const Any &specificParam) const override { return 1; };
    int64_t GetOutputNum(const Any &specificParam) const override { return 1; };
    Kernel *GetBestKernel(const LaunchParam &launchParam) const override { return nullptr; }

protected:
    Status InferShapeImpl(const LaunchParam &launchParam, SVector<Tensor> &outTensors) const override
    {
        outTensors[0].desc = launchParam.GetOutTensor(0).desc;
        return Status::OkStatus();
    }
};

TEST(OperationBaseTest, all)
{
    Mki::Operation *op = new OperationNew();
    std::unique_ptr<Kernel> kernel{nullptr};
    LaunchParam launchParam;
    Tensor inTensor;
    Tensor outTensor = inTensor;
    launchParam.AddInTensor(inTensor);
    launchParam.AddOutTensor(outTensor);

    kernel.reset(op->GetBestKernel(launchParam));
    EXPECT_EQ(kernel, nullptr);
    EXPECT_EQ(op->IsConsistent(launchParam), true);
    delete op;
    op = nullptr;
}

TEST(OperationBaseTest, all0)
{
    Mki::Operation *op = new OperationNew();
    const KernelList &kernelList = op->GetKernelList();
    EXPECT_EQ(kernelList.size(), 0);
    delete op;
    op = nullptr;
}
} // namespace
