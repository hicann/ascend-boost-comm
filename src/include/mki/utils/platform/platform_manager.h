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

#include "platform/platform_infos_def.h" // in metadef

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

    uint32_t GetPlatformInfos(const std::string socVersion, fe::PlatFormInfos &platformInfo);

private:
    PlatformManager();
    ~PlatformManager();

    uint32_t LoadConfigFile(std::string filePath);
    uint32_t LoadIniFile(std::string realFilePath);

    std::string GetSoFilePath() const;
    std::string RealPath(const std::string &filePath) const;

    void ParseVersion(PlatMapType &versionMap, std::string &socVersion) const;
    void ParsePlatformRes(const std::string &label, PlatMapType &platformResMap,
                          fe::PlatFormInfos &platInfoTemp) const;
    uint32_t ParsePlatformInfo(std::map<std::string, PlatMapType> &contentMap,
                               std::string &socVersion, fe::PlatFormInfos &platInfoTemp) const;
    bool ParseAICoreintrinsicDtypeMap(PlatMapType &aiCoreintrinsicDtypeMap,
                                      fe::PlatFormInfos &platInfoTemp) const;
    bool ParseVectorCoreintrinsicDtypeMap(PlatMapType &vectorCoreintrinsicDtypeMap,
                                          fe::PlatFormInfos &platInfoTemp) const;

    uint32_t AssemblePlatformInfoVector(std::map<std::string, PlatMapType> &contentMap);
    void FillupFixPipeInfo(fe::PlatFormInfos &platformInfos) const;

private:
    bool initFlag_;
    std::map<std::string, fe::PlatFormInfos> platformInfosMap_;
};
} // namespace Mki

#endif
