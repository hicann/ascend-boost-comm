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

#include "platform_infos_impl.h"

namespace fe {
std::map<std::string, std::vector<std::string>> PlatFormInfosImpl::GetAICoreIntrinsicDtype()
{
    return aiCoreIntrinsicDtypeMap_;
}

std::map<std::string, std::vector<std::string>> PlatFormInfosImpl::GetVectorCoreIntrinsicDtype()
{
    return vectorCoreIntrinsicDtypeMap_;
}

bool PlatFormInfosImpl::GetPlatformRes(const std::string &label, const std::string &key, std::string &value)
{
    const auto itLabel = platformResMap_.find(label);
    if (itLabel == platformResMap_.cend()) {
        return false;
    }

    auto itKey = itLabel->second.find(key);
    if (itKey == itLabel->second.end()) {
        return false;
    }

    value = itKey->second;
    return true;
}

bool PlatFormInfosImpl::GetPlatformRes(const std::string &label, std::map<std::string, std::string> &result)
{
    auto itLabel = platformResMap_.find(label);
    if (itLabel == platformResMap_.end()) {
        return false;
    }

    result = itLabel->second;
    return true;
}

void PlatFormInfosImpl::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &dtypes)
{
    aiCoreIntrinsicDtypeMap_ = dtypes;
}

void PlatFormInfosImpl::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &dtypes)
{
    vectorCoreIntrinsicDtypeMap_ = dtypes;
}

void PlatFormInfosImpl::SetPlatformRes(const std::string &label, std::map<std::string, std::string> &result)
{
    platformResMap_[label] = result;
}

void PlatFormInfosImpl::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &dtypeMap)
{
    fixpipeDtypeMap_ = dtypeMap;
}

std::map<std::string, std::vector<std::string>> PlatFormInfosImpl::GetFixPipeDtypeMap() { return fixpipeDtypeMap_; }
} // namespace fe
