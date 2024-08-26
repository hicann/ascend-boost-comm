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

#include "platform/platform_infos_def.h"
#include <mutex>
#include "mki/utils/log/log.h"
#include "platform_infos_impl.h"

namespace fe {
constexpr uint32_t MAX_CORE_NUM = 128;
bool PlatFormInfos::Init()
{
    platform_infos_impl_ = std::make_shared<PlatFormInfosImpl>();
    if (platform_infos_impl_ == nullptr) {
        return false;
    }
    return true;
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetAICoreIntrinsicDtype()
{
    return platform_infos_impl_->GetAICoreIntrinsicDtype();
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetVectorCoreIntrinsicDtype()
{
    return platform_infos_impl_->GetVectorCoreIntrinsicDtype();
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    return platform_infos_impl_->GetPlatformRes(label, key, val);
}

bool PlatFormInfos::GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    return platform_infos_impl_->GetPlatformRes(label, res);
}

void PlatFormInfos::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    platform_infos_impl_->SetAICoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    platform_infos_impl_->SetVectorCoreIntrinsicDtype(intrinsicDtypes);
}

void PlatFormInfos::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap)
{
    platform_infos_impl_->SetFixPipeDtypeMap(fixpipeDtypeMap);
}

std::mutex g_coreNumMutex;
void PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type)
{
    std::lock_guard<std::mutex> lockGuard(g_coreNumMutex);
    std::string coreNumStr;
    std::string coreTypeStr;
    if (core_type == "VectorCore") {
        coreTypeStr = "vector_core_cnt";
    } else {
        coreTypeStr = "ai_core_cnt";
    }
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Set PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
    if (coreNumStr.empty()) {
        core_num_ = 1;
        MKI_LOG(ERROR) << "CoreNumStr is empty!";
    } else {
        core_num_ = std::strtoul(coreNumStr.c_str(), nullptr, 10); // 10 进制
        if (core_num_ > MAX_CORE_NUM) {
            core_num_ = 1;
            MKI_LOG(ERROR) << "core_num is out of range : " << core_num_;
        }
    }
}

uint32_t PlatFormInfos::GetCoreNumByType(const std::string &coreType)
{
    std::lock_guard<std::mutex> lockGuard(g_coreNumMutex);
    std::string coreNumStr;
    std::string coreTypeStr = coreType == "VectorCore" ? "vector_core_cnt" : "ai_core_cnt";
    (void)GetPlatformRes("SoCInfo", coreTypeStr, coreNumStr);
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_ to " << coreTypeStr << ": " << coreNumStr;
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

uint32_t PlatFormInfos::GetCoreNum() const
{
    MKI_LOG(DEBUG) << "Get PlatFormInfos::core_num_: " << core_num_;
    return core_num_;
}

void PlatFormInfos::GetLocalMemSize(const LocalMemType &memType, uint64_t &size)
{
    std::string sizeStr;
    switch (memType) {
        case LocalMemType::L0_A: {
            (void)GetPlatformRes("AICoreSpec", "l0_a_size", sizeStr);
            break;
        }
        case LocalMemType::L0_B: {
            (void)GetPlatformRes("AICoreSpec", "l0_b_size", sizeStr);
            break;
        }
        case LocalMemType::L0_C: {
            (void)GetPlatformRes("AICoreSpec", "l0_c_size", sizeStr);
            break;
        }
        case LocalMemType::L1: {
            (void)GetPlatformRes("AICoreSpec", "l1_size", sizeStr);
            break;
        }
        case LocalMemType::L2: {
            (void)GetPlatformRes("SoCInfo", "l2_size", sizeStr);
            break;
        }
        case LocalMemType::UB: {
            (void)GetPlatformRes("AICoreSpec", "ub_size", sizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("SoCInfo", "memory_size", sizeStr);
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

void PlatFormInfos::GetLocalMemBw(const LocalMemType &memType, uint64_t &bwSize)
{
    std::string bwSizeStr;
    switch (memType) {
        case LocalMemType::L2: {
            (void)GetPlatformRes("AICoreMemoryRates", "l2_rate", bwSizeStr);
            break;
        }
        case LocalMemType::HBM: {
            (void)GetPlatformRes("AICoreMemoryRates", "ddr_rate", bwSizeStr);
            break;
        }
        default: {
            break;
        }
    }

    if (bwSizeStr.empty()) {
        bwSize = 0;
    } else {
        try {
            bwSize = static_cast<uint64_t>(std::stoll(bwSizeStr.c_str()));
        } catch (const std::invalid_argument &e) {
            bwSize = 0;
        } catch (const std::out_of_range &e) {
            bwSize = 0;
        }
    }
}

std::map<std::string, std::vector<std::string>> PlatFormInfos::GetFixPipeDtypeMap()
{
    return platform_infos_impl_->GetFixPipeDtypeMap();
}

void PlatFormInfos::SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    platform_infos_impl_->SetPlatformRes(label, res);
}
} // namespace fe
