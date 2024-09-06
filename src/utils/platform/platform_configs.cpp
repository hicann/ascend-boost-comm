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

#include "mki/utils/platform/platform_configs.h"
#include <mutex>
#include "mki/utils/log/log.h"

namespace Mki {
constexpr uint32_t MAX_CORE_NUM = 128;
void PlatformConfigs::SetPlatformSpec(const std::string &label, std::map<std::string, std::string> &res)
{
    platformSpecMap_[label] = res;
}

bool PlatformConfigs::GetPlatformSpec(const std::string &label, const std::string &key, std::string &value)
{
    const auto itLabel = platformSpecMap_.find(label);
    if (itLabel == platformSpecMap_.cend()) {
        return false;
    }

    auto itKey = itLabel->second.find(key);
    if (itKey == itLabel->second.end()) {
        return false;
    }

    value = itKey->second;
    return true;
}

bool PlatformConfigs::GetPlatformSpec(const std::string &label, std::map<std::string, std::string> &res)
{
    auto itLabel = platformSpecMap_.find(label);
    if (itLabel == platformSpecMap_.end()) {
        return false;
    }

    res = itLabel->second;
    return true;
}

uint32_t PlatformConfigs::GetCoreNumByType(const std::string &coreType)
{
    std::string coreNumStr;
    std::string coreTypeStr = coreType == "VectorCore" ? "vector_core_cnt" : "ai_core_cnt";
    (void)GetPlatformSpec("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Get PlatformConfigs::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
        return 1;
    } else {
        uint32_t coreTypeNum = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
        if (coreTypeNum > MAX_CORE_NUM) {
            MKI_LOG(ERROR) << "core_num is out of range : " << coreTypeNum;
            return 1;
        }
        return coreTypeNum;
    }
}

void PlatformConfigs::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap)
{
    fixpipeDtypeMap_ = fixpipeDtypeMap;
}

void PlatformConfigs::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    aiCoreIntrinsicDtypeMap_ = intrinsicDtypes;
}

void PlatformConfigs::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    vectorCoreIntrinsicDtypeMap_ = intrinsicDtypes;
}

const std::map<std::string, std::vector<std::string>> &PlatformConfigs::GetFixPipeDtypeMap()
{
    return fixpipeDtypeMap_;
}

std::map<std::string, std::vector<std::string>> &PlatformConfigs::GetAICoreIntrinsicDtype()
{
    return aiCoreIntrinsicDtypeMap_;
}

std::map<std::string, std::vector<std::string>> &PlatformConfigs::GetVectorCoreIntrinsicDtype()
{
    return vectorCoreIntrinsicDtypeMap_;
}

const std::map<std::string, std::map<std::string, std::string>> &PlatformConfigs::GetPlatformSpecMap()
{
    return platformSpecMap_;
}

void PlatformConfigs::GetLocalMemSize(const LocalMemType &memType, uint64_t &size)
{
    std::string sizeStr;
    switch (memType) {
        case LocalMemType::L0_A: {
            (void)GetPlatformSpec("AICoreSpec", "l0_a_size", sizeStr);
            break;
        }
        case LocalMemType::L0_B: {
            (void)GetPlatformSpec("AICoreSpec", "l0_b_size", sizeStr);
            break;
        }
        case LocalMemType::L0_C: {
            (void)GetPlatformSpec("AICoreSpec", "l0_c_size", sizeStr);
            break;
        }
        case LocalMemType::L1: {
            (void)GetPlatformSpec("AICoreSpec", "l1_size", sizeStr);
            break;
        }
        case LocalMemType::L2: {
            (void)GetPlatformSpec("SoCInfo", "l2_size", sizeStr);
            break;
        }
        case LocalMemType::UB: {
            (void)GetPlatformSpec("AICoreSpec", "ub_size", sizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformSpec("SoCInfo", "memory_size", sizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (sizeStr.empty()) {
        size = 0;
    } else {
        try {
            size = static_cast<uint64_t>(std::stoll(sizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            size = 0;
        } catch (const std::out_of_range &e) {
            size = 0;
        }
    }
}
} // namespace Mki