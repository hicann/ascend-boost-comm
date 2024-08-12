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
#include "mki/utils/cmd/cmd.h"

namespace {
using namespace Mki;

Status CmdPackage(const std::vector<std::string> &args) { return Status::OkStatus(); }

Status CmdInspect(const std::vector<std::string> &args) { return Status::OkStatus(); }

TEST(CmdTest, all)
{
    Cmd cmd;
    cmd.Register("package", std::bind(&CmdPackage, std::placeholders::_1), "package SrcOpDir DestBinFilePath");
    cmd.Register("inspect", std::bind(&CmdInspect, std::placeholders::_1), "inspect BinFilePath");
}

TEST(StatusCmdTest, Run)
{
    Cmd cmd;
    cmd.Register("package", std::bind(&CmdPackage, std::placeholders::_1), "package SrcOpDir DestBinFilePath");
    cmd.Register("inspect", std::bind(&CmdInspect, std::placeholders::_1), "inspect BinFilePath");
    const char* argv1[] = {"program"};
    EXPECT_EQ(cmd.Run(1, argv1).Message(), "invalid args");
    const char* argv2[] = {"program", "cmd3", "arg1", "arg2"};
    EXPECT_EQ(cmd.Run(4, argv2).Message(), "not support cmd:cmd3");
    const char* argv3[] = {"program", "package", "cd", "arg2"};
    Status result = cmd.Run(4, argv3);
    EXPECT_TRUE(result.Ok());
}

TEST(StatusCmdTest, DoHelp)
{
    Cmd cmd;
    const char* argv1[] = {"program", "help", "arg1", "arg2"};
    Status result = cmd.Run(4, argv1);
    EXPECT_TRUE(result.Ok());
}
} // namespace
