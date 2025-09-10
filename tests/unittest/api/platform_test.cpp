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
#include "mki/utils/log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "mki/utils/platform/platform_manager.h"
#include "acl/acl_rt.h"

namespace Mki {
TEST(PlatformTest, platformTest1)
{
    const uint32_t PLATFORM_SUCCESS = 0;
    Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
    uint32_t ret = platformManager.InitializePlatformManager();
    EXPECT_EQ(ret, PLATFORM_SUCCESS);
    PlatformConfigs platformConfigs;
    ret = platformManager.GetPlatformConfigs("Ascend910A", platformConfigs);
    EXPECT_EQ(ret, PLATFORM_SUCCESS);
    std::string val;
    platformConfigs.GetPlatformSpec("SoCInfo", "ai_core_cnt", val);
    EXPECT_EQ(val, "32");
}

TEST(PlatformInfoTest, SupportL0c2out) {
    if (Mki::PlatformInfo::Instance().GetPlatformType() != Mki::PlatformType::ASCEND_910B) {
        MKI_LOG(WARN) << "check device, in 910b, skip testcase";
        return;
    }
    Mki::PlatformInfo& platformInfo = Mki::PlatformInfo::PlatformInfo::Instance();
    bool result = platformInfo.SupportL0c2out();
    ASSERT_TRUE(result);
}

TEST(PlatformInfoTest, SupportL0c2out0) {
    PlatformConfigs platformConfigs;
    std::string label = "platform1";
    uint64_t size = 10;
    uint64_t size1 = 0;
    std::map<std::string, std::string> res;
    platformConfigs.GetPlatformSpec(label, res);
    platformConfigs.GetLocalMemSize(LocalMemType::HBM, size);
    platformConfigs.GetLocalMemSize(LocalMemType::HBM, size1);
    platformConfigs.GetFixPipeDtypeMap();
}

TEST(PlatFormInfosImplTest, GetPlatformSpec)
{
    std::string label = "platform1";
    PlatformConfigs platformConfigs;
    std::map<std::string, std::string> res;
    ASSERT_FALSE(platformConfigs.GetPlatformSpec(label, res));
    std::map<std::string, std::vector<std::string>> map1;
    EXPECT_EQ(platformConfigs.GetFixPipeDtypeMap(), map1);
}

TEST(PlatformManagerTest, Finalize)
{
    Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
    EXPECT_EQ(platformManager.Finalize(), 0);
}

TEST(PlatformInfoTest, GetCoreNumVector)
{
    uint32_t deviceId = 0;
    aclError setRes = aclrtSetDevice(deviceId);
    EXPECT_EQ(setRes, 0);
    aclrtStream stream;
    aclError createRes = aclrtCreateStream(&stream);
    EXPECT_EQ(createRes, 0);
    aclError setLimitRes = aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, 2);
    EXPECT_EQ(setLimitRes, 0);
    aclError useRes = aclrtUseStreamResInCurrentThread(stream);
    EXPECT_EQ(useRes, 0);
    uint32_t coreNum = Mki::PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    EXPECT_EQ(coreNum, 2);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
}

TEST(PlatformInfoTest, GetCoreNumCube)
{
    uint32_t deviceId = 0;
    aclError setRes = aclrtSetDevice(deviceId);
    EXPECT_EQ(setRes, 0);
    aclrtStream stream;
    aclError createRes = aclrtCreateStream(&stream);
    EXPECT_EQ(createRes, 0);
    aclError setLimitRes = aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, 2);
    EXPECT_EQ(setLimitRes, 0);
    aclError useRes = aclrtUseStreamResInCurrentThread(stream);
    EXPECT_EQ(useRes, 0);
    uint32_t coreNum = Mki::PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    EXPECT_EQ(coreNum, 2);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
}

TEST(PlatformConfigTest, GetCoreNumVector)
{
    uint32_t deviceId = 0;
    aclError setRes = aclrtSetDevice(deviceId);
    EXPECT_EQ(setRes, 0);
    aclrtStream stream;
    aclError createRes = aclrtCreateStream(&stream);
    EXPECT_EQ(createRes, 0);
    aclError setLimitRes = aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, 2);
    EXPECT_EQ(setLimitRes, 0);
    aclError useRes = aclrtUseStreamResInCurrentThread(stream);
    EXPECT_EQ(useRes, 0);
    Mki::PlatformConfigs configObj;
    uint32_t coreNum = configObj.GetCoreNumByType("VectorCore");
    EXPECT_EQ(coreNum, 2);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
}

TEST(PlatformConfigTest, GetCoreNumCube)
{
    uint32_t deviceId = 0;
    aclError setRes = aclrtSetDevice(deviceId);
    EXPECT_EQ(setRes, 0);
    aclrtStream stream;
    aclError createRes = aclrtCreateStream(&stream);
    EXPECT_EQ(createRes, 0);
    aclError setLimitRes = aclrtSetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, 2);
    EXPECT_EQ(setLimitRes, 0);
    aclError useRes = aclrtUseStreamResInCurrentThread(stream);
    EXPECT_EQ(useRes, 0);
    Mki::PlatformConfigs configObj;
    uint32_t coreNum = configObj.GetCoreNumByType("AiCore");
    EXPECT_EQ(coreNum, 2);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
}
} // namespace Mki
