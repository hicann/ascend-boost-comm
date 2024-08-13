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
#include "mki/utils/rt/device/device.h"
#include "mki/utils/log/log.h"

namespace Mki {
TEST(RtDeviceTest, RtDeviceSetSocVersionTest)
{
    const char *setVersion = "Ascend910A";
    int status = MkiRtDeviceSetSocVersion(setVersion);
    EXPECT_EQ(status, MKIRT_SUCCESS);
}

TEST(RtDeviceTest, RtDeviceGetSocVersionTest)
{
    char getVersion[100];
    uint32_t maxLen = 100;
    int status = MkiRtDeviceGetSocVersion(getVersion, maxLen);
    EXPECT_EQ(status, MKIRT_SUCCESS);
    MKI_LOG(INFO) << "DeviceVersion: " << getVersion;
}
} // namespace Mki
