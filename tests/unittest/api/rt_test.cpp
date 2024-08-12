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
#include "mki/utils/rt/rt.h"

namespace Mki {
constexpr int DEV_COUNT = 56;

TEST(RtTest, Base)
{
    int major = 0;
    int minor = 0;
    int patch = 0;
    int &major1 = major;
    int &minor1 = minor;
    int &patch1 = patch;
    MkiRtGetVersion(major1, minor1, patch1);
    EXPECT_NE(major1, 0);
    std::cout << MkiRtGetBuildInfo();
    std::cout << MkiRtErrorName(MKIRT_ERROR_NOT_INITIALIZED);
    std::cout << MkiRtErrorDesc(MKIRT_ERROR_NOT_INITIALIZED);
    std::cout << MkiRtErrorName(MKIRT_ERROR_NOT_IMPLMENT);
    std::cout << MkiRtErrorDesc(MKIRT_ERROR_NOT_IMPLMENT);
}

TEST(Rt, Device)
{
    int32_t devCount = 0;
    MkiRtDeviceGetCount(&devCount);
    int32_t devId = 0;
    MkiRtDeviceGetCurrent(&devId);
    int32_t devIds[DEV_COUNT];
    MkiRtDeviceGetIds(devIds, DEV_COUNT);
}

TEST(Rt, Mem)
{
    ASSERT_EQ(MkiRtDeviceSetCurrent(0), 0);
    void *hostPtr = nullptr;
    uint64_t size = 10;
    MkiRtMemMallocHost(&hostPtr, size);
    ASSERT_NE(hostPtr, nullptr);
    MkiRtMemFreeHost(hostPtr);
    ASSERT_EQ(MkiRtDeviceResetCurrent(0), 0);
}

TEST(Rt, Stream)
{
    ASSERT_EQ(MkiRtDeviceSetCurrent(0), 0);
    MkiRtStream stream = nullptr;
    MkiRtStreamCreate(&stream, 0);
    ASSERT_NE(stream, nullptr);
    int32_t streamId = 0;
    ASSERT_EQ(MkiRtStreamGetId(stream, &streamId), 0);
    std::cout << "streamId: " << streamId << std::endl;
    MkiRtStreamDestroy(stream);
    ASSERT_EQ(MkiRtDeviceResetCurrent(0), 0);
}
} // namespace Mki
