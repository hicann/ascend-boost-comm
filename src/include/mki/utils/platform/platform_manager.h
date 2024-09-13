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
#ifndef MKI_UTILS_PLATFORM_PLATFORM_MANAGER_H
#define MKI_UTILS_PLATFORM_PLATFORM_MANAGER_H

#include <map>
#include <string>
#include "platform_configs.h"

namespace Mki {

const uint32_t PLATFORM_FAILED = 0XFFFFFFFF;
const uint32_t PLATFORM_SUCCESS = 0;

class PlatformManager {
using PlatMapType = std::map<std::string, std::string>;

public:
    PlatformManager(const PlatformManager &) = delete;
    PlatformManager &operator=(const PlatformManager &) = delete;

    static PlatformManager &Instance();
    uint32_t InitializePlatformManager();
    uint32_t Finalize();

    uint32_t GetPlatformConfigs(const std::string socVersion, PlatformConfigs &platformConfigs);

private:
    PlatformManager();
    ~PlatformManager();

    uint32_t LoadConfigFile(std::string filePath);
    uint32_t LoadIniFile(std::string iniFileRealPath);

    void ParseVersion(PlatMapType &versionMap, std::string &socVersion) const;
    void ParsePlatformSpec(const std::string &label, PlatMapType &platformSpecMap,
                          PlatformConfigs &platformConfigsTemp) const;
    uint32_t ParsePlatformInfo(std::map<std::string, PlatMapType> &contentMap,
                               std::string &socVersion, PlatformConfigs &platformConfigsTemp) const;
    bool ParseAICoreintrinsicDtypeMap(PlatMapType &aiCoreintrinsicDtypeMap,
                                      PlatformConfigs &platformConfigsTemp) const;
    bool ParseVectorCoreintrinsicDtypeMap(PlatMapType &vectorCoreintrinsicDtypeMap,
                                          PlatformConfigs &platformConfigsTemp) const;

    uint32_t AssemblePlatformInfoVector(std::map<std::string, PlatMapType> &contentMap);
    void FillupFixPipeInfo(PlatformConfigs &platformConfigs) const;

private:
    bool initFlag_;
    std::map<std::string, PlatformConfigs> platformConfigsMap_;
};
} // namespace Mki

#endif
