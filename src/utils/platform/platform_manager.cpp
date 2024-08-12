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
#include "mki/utils/platform/platform_manager.h"
#include <cstdlib>
#include <dirent.h>
#include <dlfcn.h>
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <fstream>
#include <mutex>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/file_system/file_system.h"
#include "mki/utils/inifile/ini_file.h"

namespace Mki {
std::mutex g_pcLock;
const std::string FIXPIPE_CONFIG_KEY = "Intrinsic_fix_pipe_";
const std::string PLATFORM_RELATIVE_PATH = "/configs/platform_configs";
const std::string INI_FILE_SUFFIX = "ini";
const std::string STR_SOC_VERSION = "SoC_version";
const std::string STR_AIC_VERSION = "AIC_version";
const std::string SHORT_SOC_VERSION = "Short_SoC_version";
const std::string STR_CCEC_AIC_VERSION = "CCEC_AIC_version";
const std::string STR_CCEC_AIV_VERSION = "CCEC_AIV_version";
const std::string STR_IS_SUPPORT_AICPU = "Compiler_aicpu_support_os";
const std::string STR_VERSION = "version";
const std::string STR_SOC_INFO = "SoCInfo";
const std::string STR_AI_CORE_SPEC = "AICoreSpec";
const std::string STR_AI_CORE_MEMORY_RATES = "AICoreMemoryRates";
const std::string STR_CPU_CACHE = "CPUCache";

const std::string STR_AI_CORE_INTRINSIC_DTYPE_MAP = "AICoreintrinsicDtypeMap";
const std::string STR_VECTOR_CORE_SPEC = "VectorCoreSpec";
const std::string STR_VECTOR_CORE_MEMORY_RATES = "VectorCoreMemoryRates";
const std::string STR_VECTOR_CORE_INTRINSIC_DTYPE_MAP = "VectorCoreintrinsicDtypeMap";

const std::string SOC_VERSION_ASCEND910 = "Ascend910";
const std::string SOC_VERSION_ASCEND910A = "Ascend910A";
const std::string SOC_VERSION_ASCEND320 = "Ascend320";
const std::string SOC_VERSION_ASCEND031 = "Ascend031";

const std::string SOC_VERSION_ASCEND310B1 = "Ascend310B1";
const std::string SOC_VERSION_ASCEND910B1 = "Ascend910B1";
const std::string SOC_VERSION_ASCEND910B2 = "Ascend910B2";
const std::string SOC_VERSION_ASCEND910B3 = "Ascend910B3";
const std::string SOC_VERSION_ASCEND910B4 = "Ascend910B4";
const std::string SOC_VERSION_ASCEND910C1 = "Ascend910C1";
const std::string SOC_VERSION_ASCEND910C2 = "Ascend910C2";
const std::string SOC_VERSION_ASCEND910C3 = "Ascend910C3";
const std::string SOC_VERSION_ASCEND910C4 = "Ascend910C4";

const std::unordered_set<std::string> FIXPIPE_SOCVERSION = {
    SOC_VERSION_ASCEND320,   SOC_VERSION_ASCEND310B1, SOC_VERSION_ASCEND910B1, SOC_VERSION_ASCEND910B2,
    SOC_VERSION_ASCEND910B3, SOC_VERSION_ASCEND910B4, SOC_VERSION_ASCEND910C1, SOC_VERSION_ASCEND910C2,
    SOC_VERSION_ASCEND910C3, SOC_VERSION_ASCEND910C4, SOC_VERSION_ASCEND031
};

enum class PlatformInfoType {
    EN_VERSION = 0,
    EN_SOC_INFO,
    EN_AI_CORE_SPEC,
    EN_AI_CORE_MEMORY_RATES,
    EN_CPU_CACHE,
    EN_AI_CORE_INTRINSIC_DTYPE_MAP,
    EN_VECOTR_CORE_SPEC,
    EN_VECOTR_CORE_MEMORY_RATES,
    EN_VECOTR_CORE_INTRINSIC_DTYPE_MAP,
};

std::map<std::string, PlatformInfoType> g_platformInfoTypeMap = {
    {STR_VERSION, PlatformInfoType::EN_VERSION},
    {STR_SOC_INFO, PlatformInfoType::EN_SOC_INFO},
    {STR_AI_CORE_SPEC, PlatformInfoType::EN_AI_CORE_SPEC},
    {STR_AI_CORE_MEMORY_RATES, PlatformInfoType::EN_AI_CORE_MEMORY_RATES},
    {STR_CPU_CACHE, PlatformInfoType::EN_CPU_CACHE},
    {STR_AI_CORE_INTRINSIC_DTYPE_MAP, PlatformInfoType::EN_AI_CORE_INTRINSIC_DTYPE_MAP},
    {STR_VECTOR_CORE_SPEC, PlatformInfoType::EN_VECOTR_CORE_SPEC},
    {STR_VECTOR_CORE_MEMORY_RATES, PlatformInfoType::EN_VECOTR_CORE_MEMORY_RATES},
    {STR_VECTOR_CORE_INTRINSIC_DTYPE_MAP, PlatformInfoType::EN_VECOTR_CORE_INTRINSIC_DTYPE_MAP}};

PlatformManager::PlatformManager() : initFlag_(false) {}

PlatformManager::~PlatformManager() {}

PlatformManager &PlatformManager::Instance()
{
    static PlatformManager platformInfo;
    return platformInfo;
}

bool IsSpace(char c)
{
    if (c == ' ' || c == '\t') {
        return true;
    }
    return false;
}

uint32_t PlatformManager::LoadIniFile(std::string iniFileRealPath)
{
    std::map<std::string, std::map<std::string, std::string>> contentInfoMap;
    bool isParseSuccess = IniFile::ParseIniFileToMap(iniFileRealPath, contentInfoMap);
    MKI_CHECK(isParseSuccess, "ParseIniFileToMap failed.", return PLATFORM_FAILED);
    MKI_CHECK(AssemblePlatformInfoVector(contentInfoMap) == PLATFORM_SUCCESS, "Assemble platform info failed.",
        return PLATFORM_FAILED);

    return PLATFORM_SUCCESS;
}

uint32_t PlatformManager::LoadConfigFile(std::string filePath)
{
    std::vector<std::string> iniCfgFiles;
    DIR *dir;
    struct dirent *dirp = nullptr;
    const char *fileSuffix = ".ini";
    std::string realPath = FileSystem::PathCheckAndRegular(filePath, false);
    dir = opendir(realPath.c_str());
    MKI_CHECK(dir != nullptr, "Fail to open directory " << realPath.c_str(), return PLATFORM_FAILED);

    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_name[0] == '.') {
            continue;
        }
        std::string fileName = dirp->d_name;
        if (strlen(dirp->d_name) <= strlen(fileSuffix)) {
            continue;
        }
        size_t pos = fileName.rfind('.');
        if (pos == std::string::npos || fileName.substr(pos + 1) != INI_FILE_SUFFIX) {
            continue;
        }
        if (iniCfgFiles.size() >= MAX_FILE_NUM) {
            MKI_LOG(ERROR) << "Max file number exceeded ";
            closedir(dir);
            return PLATFORM_FAILED;
        }
        if (strcmp(&(dirp->d_name)[strlen(dirp->d_name) - strlen(fileSuffix)], fileSuffix) == 0) {
            iniCfgFiles.push_back(realPath + "/" + dirp->d_name);
        }
    }
    closedir(dir);

    MKI_CHECK(!iniCfgFiles.empty(), "there is no ini file in path " << realPath.c_str(), return PLATFORM_FAILED);

    for (std::string iniFilePath : iniCfgFiles) {
        MKI_CHECK(LoadIniFile(iniFilePath) == PLATFORM_SUCCESS, "Fail to load ini file " << iniFilePath.c_str(),
            return PLATFORM_FAILED);
    }
    return PLATFORM_SUCCESS;
}

void PlatformManager::ParseVersion(std::map<std::string, std::string> &versionMap, std::string &socVersion) const
{
    std::map<std::string, std::string>::const_iterator soc = versionMap.find(STR_SOC_VERSION);
    if (soc != versionMap.end()) {
        socVersion = soc->second;
    }
}

static std::vector<std::string> Split(const std::string &str, char pattern)
{
    std::vector<std::string> resVec;
    if (str.empty()) {
        return resVec;
    }

    std::string strAndPattern = str + pattern;
    size_t pos = strAndPattern.find(pattern);
    size_t size = strAndPattern.size();
    while (pos != std::string::npos) {
        std::string subStr = strAndPattern.substr(0, pos);
        resVec.push_back(subStr);
        strAndPattern = strAndPattern.substr(pos + 1, size);
        pos = strAndPattern.find(pattern);
    }
    return resVec;
}

uint32_t PlatformManager::AssemblePlatformInfoVector(
    std::map<std::string, std::map<std::string, std::string>> &contentInfoMap)
{
    std::string socVersion;
    fe::PlatFormInfos platformInfos;
    if (!platformInfos.Init()) {
        return PLATFORM_FAILED;
    }

    if (ParsePlatformInfo(contentInfoMap, socVersion, platformInfos) != PLATFORM_SUCCESS) {
        MKI_LOG(ERROR) << "Parse platform info from content failed";
        return PLATFORM_FAILED;
    }
    if (FIXPIPE_SOCVERSION.count(socVersion) != 0) {
        FillupFixPipeInfo(platformInfos);
    }
    if (!socVersion.empty()) {
        auto iter = platformInfosMap_.find(socVersion);
        if (iter == platformInfosMap_.end()) {
            platformInfosMap_.emplace(make_pair(socVersion, platformInfos));
        } else {
            MKI_LOG(ERROR) << "There are repetitive soc version[" << socVersion.c_str() << "] in config files";
            return PLATFORM_FAILED;
        }
    }

    return PLATFORM_SUCCESS;
}

void PlatformManager::FillupFixPipeInfo(fe::PlatFormInfos &platformInfos) const
{
    std::map<std::string, std::vector<std::string>> aicoreMap = platformInfos.GetAICoreIntrinsicDtype();
    std::map<std::string, std::vector<std::string>> fixpipeMap;
    for (auto iter = aicoreMap.begin(); iter != aicoreMap.end(); iter++) {
        if (iter->first.find(FIXPIPE_CONFIG_KEY) != iter->first.npos) {
            fixpipeMap.emplace(make_pair(iter->first, iter->second));
        }
    }
    if (fixpipeMap.empty()) {
        return;
    }
    platformInfos.SetFixPipeDtypeMap(fixpipeMap);
}

bool PlatformManager::ParseAICoreintrinsicDtypeMap(std::map<std::string, std::string> &aiCoreintrinsicDtypeMap,
    fe::PlatFormInfos &platformInfoTemp) const
{
    std::map<std::string, std::string>::const_iterator iter;
    std::map<std::string, std::vector<std::string>> aicoreIntrinsicDtypes = platformInfoTemp.GetAICoreIntrinsicDtype();
    std::map<std::string, std::string> platformRes;
    for (iter = aiCoreintrinsicDtypeMap.begin(); iter != aiCoreintrinsicDtypeMap.end(); iter++) {
        size_t pos = iter->second.find('|');
        if (pos == std::string::npos) {
            return false;
        }
        std::string key = iter->second.substr(0, pos);
        std::string value = iter->second.substr(pos + 1);
        std::vector<std::string> dtypeVector = Split(value, ',');
        platformRes.emplace(make_pair(key, value));
        aicoreIntrinsicDtypes.emplace(make_pair(key, dtypeVector));
    }
    platformInfoTemp.SetPlatformRes(STR_AI_CORE_INTRINSIC_DTYPE_MAP, platformRes);
    platformInfoTemp.SetAICoreIntrinsicDtype(aicoreIntrinsicDtypes);
    return true;
}

bool PlatformManager::ParseVectorCoreintrinsicDtypeMap(std::map<std::string, std::string> &vectorCoreintrinsicDtypeMap,
    fe::PlatFormInfos &platformInfoTemp) const
{
    std::map<std::string, std::string>::const_iterator iter;
    std::map<std::string, std::vector<std::string>> vectorCoreIntrinsicDtypeMap =
        platformInfoTemp.GetVectorCoreIntrinsicDtype();
    std::map<std::string, std::string> platformRes;
    for (iter = vectorCoreintrinsicDtypeMap.begin(); iter != vectorCoreintrinsicDtypeMap.end(); iter++) {
        size_t pos = iter->second.find('|');
        if (pos == std::string::npos) {
            MKI_LOG(ERROR) << "The intrinsic_dtype_map string does not contain |";
            return false;
        }
        std::string key = iter->second.substr(0, pos);
        std::string value = iter->second.substr(pos + 1);
        std::vector<std::string> dtypeVector = Split(value, ',');
        platformRes.emplace(make_pair(key, value));
        vectorCoreIntrinsicDtypeMap.emplace(make_pair(key, dtypeVector));
    }
    platformInfoTemp.SetPlatformRes(STR_VECTOR_CORE_INTRINSIC_DTYPE_MAP, platformRes);
    platformInfoTemp.SetVectorCoreIntrinsicDtype(vectorCoreIntrinsicDtypeMap);
    return true;
}

void PlatformManager::ParsePlatformRes(const std::string &label, std::map<std::string, std::string> &platformResMap,
    fe::PlatFormInfos &platformInfoTemp) const
{
    platformInfoTemp.SetPlatformRes(label, platformResMap);
}

uint32_t PlatformManager::ParsePlatformInfo(std::map<std::string, std::map<std::string, std::string>> &contentMap,
    std::string &socVersion, fe::PlatFormInfos &platformInfoTemp) const
{
    std::map<std::string, std::map<std::string, std::string>>::iterator it;
    for (it = contentMap.begin(); it != contentMap.end(); it++) {
        if (it->first == STR_VERSION) {
            ParseVersion(it->second, socVersion);
        } else if (it->first == STR_AI_CORE_INTRINSIC_DTYPE_MAP) {
            MKI_CHECK(ParseAICoreintrinsicDtypeMap(it->second, platformInfoTemp),
                "failed to parse aicore intrinsic dtype map", return PLATFORM_FAILED);
        } else if (it->first == STR_VECTOR_CORE_INTRINSIC_DTYPE_MAP) {
            MKI_CHECK(ParseVectorCoreintrinsicDtypeMap(it->second, platformInfoTemp),
                "failed to parse vectorcore intrinsic dtype map", return PLATFORM_FAILED);
        } else {
            ParsePlatformRes(it->first, it->second, platformInfoTemp);
        }
    }
    return PLATFORM_SUCCESS;
}

std::string PlatformManager::RealPath(const std::string &path) const
{
    if (path.empty()) {
        MKI_LOG(ERROR) << "path string is NULL";
        return "";
    }

    if (path.size() >= PATH_MAX) {
        MKI_LOG(ERROR) << "file path " << path.c_str() << " is too long!";
        return "";
    }

    char resolvedPath[PATH_MAX] = {0};
    std::string res = "";

    if (realpath(path.c_str(), resolvedPath) != nullptr) {
        res = resolvedPath;
    } else {
        MKI_LOG(ERROR) << "path " << path.c_str() << " is not exist";
    }
    return res;
}

std::string PlatformManager::GetSoFilePath() const
{
    Dl_info dlInfo;
    std::string realFilePath = "";
    PlatformManager &(*instancePtr)() = &PlatformManager::Instance;
    if (dladdr(reinterpret_cast<void *>(instancePtr), &dlInfo) == 0) {
        MKI_LOG(ERROR) << "Failed to read the so file path";
        return realFilePath;
    } else {
        std::string soPath = dlInfo.dli_fname;
        if (soPath.empty()) {
            MKI_LOG(ERROR) << "So file path is empty";
            return realFilePath;
        }
        realFilePath = FileSystem::PathCheckAndRegular(soPath);
        MKI_CHECK(!realFilePath.empty(), "File path is invalid", return "");
        size_t pos = realFilePath.rfind('/');
        if (pos >= SIZE_MAX) {
            MKI_LOG(ERROR) << "The path of current so file does not contain /";
            return realFilePath;
        }
        realFilePath = realFilePath.substr(0, pos + 1);
    }
    return realFilePath;
}

uint32_t PlatformManager::InitializePlatformManager()
{
    std::lock_guard<std::mutex> lockGuard(g_pcLock);
    if (initFlag_) {
        return PLATFORM_SUCCESS;
    }
    const char *mkiHomePath = getenv("MKI_HOME_PATH");
    if (mkiHomePath == nullptr) {
        MKI_LOG(ERROR) << "env MKI_HOME_PATH not exists";
        return PLATFORM_FAILED;
    }
    std::string mkiHomePathStr = mkiHomePath;
    if (mkiHomePathStr == "") {
        MKI_LOG(ERROR) << "getenv failed";
        return PLATFORM_FAILED;
    }
    std::string cfgFileRealPath = FileSystem::PathCheckAndRegular(mkiHomePathStr + PLATFORM_RELATIVE_PATH, false);
    if (cfgFileRealPath.empty()) {
        MKI_LOG(ERROR) << "File path " << cfgFileRealPath.c_str() << " is not valid";
        return PLATFORM_FAILED;
    }

    uint32_t ret = LoadConfigFile(cfgFileRealPath);
    if (ret != PLATFORM_SUCCESS) {
        MKI_LOG(ERROR) << "Load cfg file failed, path is " << cfgFileRealPath.c_str();
        return PLATFORM_FAILED;
    }
    initFlag_ = true;
    return PLATFORM_SUCCESS;
}

uint32_t PlatformManager::GetPlatformInfos(const std::string socVersion, fe::PlatFormInfos &platformInfo)
{
    std::string realSocVersion = socVersion;
    if (realSocVersion == SOC_VERSION_ASCEND910) {
        realSocVersion = SOC_VERSION_ASCEND910A;
    }
    auto iter = platformInfosMap_.find(realSocVersion);
    if (iter == platformInfosMap_.end()) {
        MKI_LOG(ERROR) << "Can not found platform_info by socVersion " << realSocVersion.c_str();
        return PLATFORM_FAILED;
    }
    platformInfo = iter->second;
    return PLATFORM_SUCCESS;
}

uint32_t PlatformManager::Finalize()
{
    std::lock_guard<std::mutex> lockGuard(g_pcLock);
    if (!initFlag_) {
        return PLATFORM_SUCCESS;
    }
    platformInfosMap_.clear();
    initFlag_ = false;
    return PLATFORM_SUCCESS;
}
} // namespace fe
