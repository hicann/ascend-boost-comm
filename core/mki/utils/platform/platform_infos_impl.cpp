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

bool PlatFormInfosImpl::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    const auto itLabel = platformResMap_.find(label);
    if (itLabel == platformResMap_.cend()) {
        return false;
    }

    auto itKey = itLabel->second.find(key);
    if (itKey == itLabel->second.end()) {
        return false;
    }

    val = itKey->second;
    return true;
}

bool PlatFormInfosImpl::GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    auto itLabel = platformResMap_.find(label);
    if (itLabel == platformResMap_.end()) {
        return false;
    }

    res = itLabel->second;
    return true;
}

void PlatFormInfosImpl::SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    aiCoreIntrinsicDtypeMap_ = intrinsicDtypes;
}

void PlatFormInfosImpl::SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes)
{
    vectorCoreIntrinsicDtypeMap_ = intrinsicDtypes;
}

void PlatFormInfosImpl::SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res)
{
    platformResMap_[label] = res;
}

void PlatFormInfosImpl::SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap)
{
    fixpipeDtypeMap_ = fixpipeDtypeMap;
}

std::map<std::string, std::vector<std::string>> PlatFormInfosImpl::GetFixPipeDtypeMap() { return fixpipeDtypeMap_; }
} // namespace fe
