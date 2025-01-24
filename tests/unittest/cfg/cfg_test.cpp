/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include "mki/utils/cfg/cfg_core.h"

namespace Mki {

TEST(TestCfg, ParseCfgIniSuccess)
{
    char buf[256];
    auto result = getcwd(buf, sizeof(buf));
    ASSERT_NE(result, nullptr);
    std::string currentPath(buf);
    std::string fileName = currentPath + "/test_cfg.ini";
    std::ofstream file(fileName);
    file << "[LOG] \n LOG_TO_STDOUT=1 \n";
    file.close();

    CfgCore cfgCore;
    Status st = cfgCore.LoadFromFile(fileName);
    cfgCore.ParseLogData();
    ASSERT_EQ(st.Ok(), true);
    EXPECT_EQ(cfgCore.GetLogCfg().isLogToStdOut, true);
    std::remove(fileName.c_str());
}

TEST(TestCfg, GetCfgIniLogItems)
{
    char buf[256];
    auto result = getcwd(buf, sizeof(buf));
    ASSERT_NE(result, nullptr);
    std::string currentPath(buf);
    std::string fileName = currentPath + "/test_cfg.ini";
    std::ofstream file(fileName);
    file << "[LOG] \n LOG_TO_STDOUT=0 \n LOG_LEVEL=DEBUG \n LOG_TO_FILE=1 \n"
            "LOG_TO_FILE_FLUSH=1 \n LOG_TO_BOOST_TYPE=test \n LOG_PATH=test1";
    file.close();

    CfgCore cfgCore;
    Status st = cfgCore.LoadFromFile(fileName);
    cfgCore.ParseLogData();
    ASSERT_EQ(st.Ok(), true);
    EXPECT_EQ(cfgCore.GetLogCfg().isLogToStdOut, false);
    EXPECT_EQ(cfgCore.GetLogCfg().logLevel, "DEBUG");
    EXPECT_EQ(cfgCore.GetLogCfg().isLogToFile, true);
    EXPECT_EQ(cfgCore.GetLogCfg().isLogToFileFlush, true);
    EXPECT_EQ(cfgCore.GetLogCfg().logToBoostType, "test");
    EXPECT_EQ(cfgCore.GetLogCfg().logPath, "test1");
    std::remove(fileName.c_str());
}
} // namespace Mki