/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <map>
#include <iostream>
#include <cstring>

#include "mki/utils/inifile/ini_file.h"
#include "mki/utils/log/log_core.h"
#include "mki/utils/cfg/cfg_core.h"

namespace Mki {
const size_t MKI_MAX_ENV_STRING_LEN = 12800;

static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("ASDOPS_LOG_TO_STDOUT");
    return envLogToStdout != nullptr && strlen(envLogToStdout) <= MKI_MAX_ENV_STRING_LEN &&
           strcmp(envLogToStdout, "1") == 0;
}

CfgCore& CfgCore::GetCfgCoreInstance()
{
    static CfgCore cfgCore;
    return cfgCore;
}

CfgCore::CfgCore()
{
    printLog_ = GetLogToStdoutFromEnv();
    InitMkiHomePath();
    LoadFromFile(mkiHomePath_ + "/configs/mki_cfg.ini");
    ParseLogData();
}

Status CfgCore::LoadFromFile(const std::string &fileName)
{
    cfgFileExixts_ = IniFile::ParseIniFileToMapNoLog(fileName, rawInfoMap_);
    if (!cfgFileExixts_ || rawInfoMap_.empty()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    return Status::OkStatus();
}

Status CfgCore::ParseLogData()
{
    std::map<std::string, std::map<std::string, std::string>>::iterator it = rawInfoMap_.find("LOG");
    if (it == rawInfoMap_.end()) {
        return Status::FailStatus(ERROR_INVALID_VALUE);
    }
    if (it->second.find("LOG_TO_STDOUT") != it->second.end()) {
        logCfgData_.isLogToStdOut = (it->second["LOG_TO_STDOUT"].compare("1") == 0);
    }
    if (it->second.find("LOG_LEVEL") != it->second.end()) {
        logCfgData_.logLevel = it->second["LOG_LEVEL"];
    }
    if (it->second.find("LOG_TO_FILE") != it->second.end()) {
        logCfgData_.isLogToFile = (it->second["LOG_TO_FILE"].compare("1") == 0);
    }
    if (it->second.find("LOG_TO_FILE_FLUSH") != it->second.end()) {
        logCfgData_.isLogToFileFlush = (it->second["LOG_TO_FILE_FLUSH"].compare("1") == 0);
    }
    if (it->second.find("LOG_TO_BOOST_TYPE") != it->second.end()) {
        logCfgData_.logToBoostType = it->second["LOG_TO_BOOST_TYPE"];
    }
    if (it->second.find("LOG_PATH") != it->second.end()) {
        logCfgData_.logPath = it->second["LOG_PATH"];
    }
    return Status::OkStatus();
}

void CfgCore::InitMkiHomePath()
{
    const char *envStr = std::getenv("MKI_HOME_PATH");
    if (!envStr) {
        return;
    }
    if (strlen(envStr) > MKI_MAX_ENV_STRING_LEN && printLog_) {
        return;
    }
    mkiHomePath_ = std::string(envStr);
}

bool CfgCore::CfgFileExists() const
{
    return cfgFileExixts_;
}

const LogCfg CfgCore::GetLogCfg() const
{
    return logCfgData_;
}
} // namespace Mki