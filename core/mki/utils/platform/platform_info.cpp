/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mki/utils/platform/platform_info.h"

#include "mki/utils/assert/assert.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/log/log.h"
#include "mki/utils/rt/rt.h"
#include "mki/utils/platform/platform_manager.h"

namespace Mki {
PlatformInfo::PlatformInfo() { Init(); }

PlatformInfo &PlatformInfo::Instance()
{
    static PlatformInfo platformInfo;
    return platformInfo;
}

void PlatformInfo::Init()
{
    if (inited_) {
        return;
    }

    const uint32_t maxLen = 100;
    char version[maxLen];
    if (MkiRtDeviceGetSocVersion(version, maxLen) != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "Failed to get soc version";
        return;
    }
    std::string socVersion(version);
    MKI_LOG(INFO) << "PlatformInfo get soc version: " << socVersion;

    if (socVersion.find("Ascend310P") == 0) {
        platformType_ = PlatformType::ASCEND_310P;
    } else if (socVersion.find("Ascend910A") == 0) {
        platformType_ = PlatformType::ASCEND_910A;
    } else if (socVersion.find("Ascend910B") == 0) {
        platformType_ = PlatformType::ASCEND_910B;
    } else if (socVersion.find("Ascend910C") == 0) {
        platformType_ = PlatformType::ASCEND_910B;
    } else {
        MKI_LOG(ERROR) << "Unsupport soc";
        return;
    }

    Mki::PlatformManager &platformManager = Mki::PlatformManager::Instance();
    MKI_CHECK(platformManager.InitializePlatformManager() == PLATFORM_SUCCESS,
                 "Initialize platform manager failed", return);
    MKI_CHECK(platformManager.GetPlatformInfos(socVersion, platformInfo_) == PLATFORM_SUCCESS,
                 "Get platform Information failed", return);
    inited_ = true;
}

bool PlatformInfo::Inited() const { return inited_; }

uint32_t PlatformInfo::GetCoreNum(CoreType type)
{
    if (platformType_ == PlatformType::ASCEND_910B) {
        switch (type) {
            case CoreType::CORE_TYPE_VECTOR:
                platformInfo_.SetCoreNumByCoreType("VectorCore");
                break;
            default:
                platformInfo_.SetCoreNumByCoreType("AiCore");
                break;
        }
    } else {
        platformInfo_.SetCoreNumByCoreType("AiCore");
    }
    uint32_t coreNum = platformInfo_.GetCoreNum();
    MKI_FLOG_INFO("Platform get core num %u, type %d", coreNum, type);
    return coreNum;
}

uint64_t PlatformInfo::GetL2Size()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::L2, memSize);
    MKI_LOG(INFO) << "PlatformInfo get l2 size " << memSize;
    return memSize;
}

uint64_t PlatformInfo::GetL1Size()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::L1, memSize);
    MKI_LOG(INFO) << "PlatformInfo get l1 size " << memSize;
    return memSize;
}

uint64_t PlatformInfo::GetL0ASize()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::L0_A, memSize);
    MKI_LOG(INFO) << "PlatformInfo get l0a size " << memSize;
    return memSize;
}

uint64_t PlatformInfo::GetL0BSize()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::L0_B, memSize);
    MKI_LOG(INFO) << "PlatformInfo get l0b size " << memSize;
    return memSize;
}

uint64_t PlatformInfo::GetL0CSize()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::L0_C, memSize);
    MKI_LOG(INFO) << "PlatformInfo get l0c size " << memSize;
    return memSize;
}

uint64_t PlatformInfo::GetUbSize()
{
    uint64_t memSize = 0;
    platformInfo_.GetLocalMemSize(fe::LocalMemType::UB, memSize);
    MKI_LOG(INFO) << "PlatformInfo get ub size " << memSize;
    return memSize;
}

bool PlatformInfo::SupportL0c2out() { return GetAicoreIntrinsic("Intrinsic_fix_pipe_l0c2out"); }

bool PlatformInfo::GetAicoreIntrinsic(const string &intrinsicName)
{
    std::string val;
    (void)platformInfo_.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsicName, val);
    if (!val.empty()) {
        MKI_LOG(INFO) << "PlatformInfo get aicore intrinsic " << intrinsicName << " = " << val;
        return true;
    } else {
        MKI_LOG(INFO) << "PlatformInfo get no aicore intrinsic " << intrinsicName;
        return false;
    }
}

PlatformType PlatformInfo::GetPlatformType() const { return platformType_; }
} // namespace Mki
