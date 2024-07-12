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
#ifndef MKI_TBE_PLATFORM_MANAGER_H
#define MKI_TBE_PLATFORM_MANAGER_H

#include "platform/platform_infos_def.h" // in metadef

namespace Mki {

const uint32_t PLATFORM_FAILED = 0XFFFFFFFF;
const uint32_t PLATFORM_SUCCESS = 0;

class PlatformManager {
public:
    PlatformManager(const PlatformManager &) = delete;
    PlatformManager &operator=(const PlatformManager &) = delete;

    static PlatformManager &Instance();
    uint32_t InitializePlatformManager();
    uint32_t Finalize();

    uint32_t GetPlatformInfos(const std::string socVersion, fe::PlatFormInfos &platformInfo);

private:
    PlatformManager();
    ~PlatformManager();

    uint32_t LoadIniFile(std::string iniFileRealPath);

    uint32_t LoadConfigFile(std::string filePath);

    std::string RealPath(const std::string &path) const;

    std::string GetSoFilePath() const;

    void ParseVersion(std::map<std::string, std::string> &versionMap, std::string &socVersion) const;

    bool ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &aiCoreintrinsicDtypeMap,
                                      fe::PlatFormInfos &platformInfoTemp) const;

    bool ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vectorCoreintrinsicDtypeMap,
                                          fe::PlatFormInfos &platformInfoTemp) const;

    void ParsePlatformRes(const std::string &label, std::map<std::string, std::string> &platformResMap,
                          fe::PlatFormInfos &platformInfoTemp) const;

    uint32_t ParsePlatformInfo(std::map<std::string, std::map<std::string, std::string>> &contentMap,
                               std::string &socVersion, fe::PlatFormInfos &platformInfoTemp) const;

    uint32_t AssemblePlatformInfoVector(std::map<std::string, std::map<std::string, std::string>> &contentInfoMap);
    void FillupFixPipeInfo(fe::PlatFormInfos &platformInfos) const;

    bool initFlag_;

    std::map<std::string, fe::PlatFormInfos> platformInfosMap_;
};
} // namespace Mki

#endif
