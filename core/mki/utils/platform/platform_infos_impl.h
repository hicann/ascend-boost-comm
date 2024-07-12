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

#ifndef MKI_PLATFORM_INFOS_IMPL_H
#define MKI_PLATFORM_INFOS_IMPL_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform/platform_infos_def.h"

namespace fe {
class PlatFormInfosImpl {
public:
    std::map<std::string, std::vector<std::string>> GetAICoreIntrinsicDtype();
    std::map<std::string, std::vector<std::string>> GetVectorCoreIntrinsicDtype();
    bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
    bool GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);

    void SetAICoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes);
    void SetVectorCoreIntrinsicDtype(std::map<std::string, std::vector<std::string>> &intrinsicDtypes);
    void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);
    std::map<std::string, std::vector<std::string>> GetFixPipeDtypeMap();
    void SetFixPipeDtypeMap(const std::map<std::string, std::vector<std::string>> &fixpipeDtypeMap);

private:
    std::map<std::string, std::vector<std::string>> aiCoreIntrinsicDtypeMap_;
    std::map<std::string, std::vector<std::string>> vectorCoreIntrinsicDtypeMap_;
    std::map<std::string, std::map<std::string, std::string>> platformResMap_;
    std::map<std::string, std::vector<std::string>> fixpipeDtypeMap_;
};
} // namespace fe

#endif
