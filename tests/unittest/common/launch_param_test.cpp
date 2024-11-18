/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "mki/launch_param.h"

namespace {
class LaunchParamTest : public ::testing::Test {
protected:
    Mki::LaunchParam launchParam;
    Mki::Tensor tensor1, tensor2;

    void SetUp() override
    {
        tensor1.desc = Mki::TensorDesc{};
        tensor1.data = malloc(128); // Allocate 128 bytes for tensor1.data
        tensor1.dataSize = 128; // Set tensor1.dataSize to 128
        tensor1.hostData = malloc(128); // Allocate 128 bytes for tensor1.hostData

        tensor2.desc = Mki::TensorDesc{}; // Initialize tensor2.desc
        tensor2.data = malloc(256); // Allocate 256 bytes for tensor2.data
        tensor2.dataSize = 256; // Set tensor2.dataSize to 256
        tensor2.hostData = malloc(256); // Allocate 256 bytes for tensor2.hostData
    }

    void TearDown() override
    {
        free(tensor1.data);
        free(tensor1.hostData);
        free(tensor2.data);
        free(tensor2.hostData);
    }
};

TEST_F(LaunchParamTest, AddInputTensor)
{
    EXPECT_EQ(launchParam.GetInTensorCount(), 0); // 验证添加的输入张量数量是否为0
    launchParam.AddInTensor(tensor1);
    EXPECT_EQ(launchParam.GetInTensorCount(), 1); // 验证添加的输入张量数量是否为1
    launchParam.AddInTensor(tensor2);
    EXPECT_EQ(launchParam.GetInTensorCount(), 2); // 验证添加的输入张量数量是否为2
}

TEST_F(LaunchParamTest, GetInputTensor)
{
    launchParam.AddInTensor(tensor1);
    launchParam.AddInTensor(tensor2);
    EXPECT_EQ(launchParam.GetInTensor(0).dataSize, tensor1.dataSize);
    EXPECT_EQ(launchParam.GetInTensor(1).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, AddOutputTensor)
{
    EXPECT_EQ(launchParam.GetOutTensorCount(), 0); // 验证输出张量数量是否为0
    launchParam.AddOutTensor(tensor1);
    EXPECT_EQ(launchParam.GetOutTensorCount(), 1); // 验证输出张量数量是否为1
    launchParam.AddOutTensor(tensor2);
    EXPECT_EQ(launchParam.GetOutTensorCount(), 2); // 验证输出张量数量是否为2
}

TEST_F(LaunchParamTest, GetOutputTensor)
{
    launchParam.AddOutTensor(tensor1);
    launchParam.AddOutTensor(tensor2);
    EXPECT_EQ(launchParam.GetOutTensor(0).dataSize, tensor1.dataSize);
    EXPECT_EQ(launchParam.GetOutTensor(1).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, Reset)
{
    launchParam.AddInTensor(tensor1);
    launchParam.AddOutTensor(tensor2);
    EXPECT_EQ(launchParam.GetInTensorCount(), 1); // 输入张量数量设置为1
    EXPECT_EQ(launchParam.GetOutTensorCount(), 1); // 输出张量数量设置为1
    launchParam.Reset();
    EXPECT_EQ(launchParam.GetInTensorCount(), 0); // 验证输入张量数量是否为0
    EXPECT_EQ(launchParam.GetOutTensorCount(), 0); // 验证输出张量数量是否为0
}

TEST_F(LaunchParamTest, AssignmentOperator)
{
    Mki::LaunchParam launchParam2;
    launchParam.AddInTensor(tensor1);
    launchParam.AddOutTensor(tensor2);
    launchParam2 = launchParam;
    EXPECT_EQ(launchParam2.GetInTensorCount(), 1); // 验证输入张量数量是否为1
    EXPECT_EQ(launchParam2.GetOutTensorCount(), 1); // 验证输出张量数量是否为1
    EXPECT_EQ(launchParam2.GetInTensor(0).dataSize, tensor1.dataSize);
    EXPECT_EQ(launchParam2.GetOutTensor(0).dataSize, tensor2.dataSize);
}

TEST_F(LaunchParamTest, ToString)
{
    launchParam.AddInTensor(tensor1);
    launchParam.AddOutTensor(tensor2);

    std::string result = launchParam.ToString();
    EXPECT_NE(result.find("intensors[0]"), std::string::npos);
    EXPECT_NE(result.find("outtensors[0]"), std::string::npos);
}
} // namespace
