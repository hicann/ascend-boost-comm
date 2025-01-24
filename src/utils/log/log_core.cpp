/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "mki/utils/env/env.h"
#include "mki/utils/cfg/cfg_core.h"
#include "mki/utils/log/log_sink_stdout.h"
#include "mki/utils/log/log_sink_file.h"
#include "mki/utils/log/log.h"
#include "mki/utils/log/log_core.h"

namespace Mki {
static bool GetLogToStdoutFromEnvCfg(bool controlledByCfg)
{
    if (controlledByCfg) {
        return CfgCore::GetCfgCoreInstance().GetLogCfg().isLogToStdOut;
    }
    const char *envLogToStdout = std::getenv("ASDOPS_LOG_TO_STDOUT");
    return envLogToStdout != nullptr && strlen(envLogToStdout) <= MAX_ENV_STRING_LEN &&
           strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnvCfg(bool controlledByCfg)
{
    if (controlledByCfg) {
        return CfgCore::GetCfgCoreInstance().GetLogCfg().isLogToFile;
    }
    const char *envLogToFile = std::getenv("ASDOPS_LOG_TO_FILE");
    return envLogToFile != nullptr && strlen(envLogToFile) <= MAX_ENV_STRING_LEN && strcmp(envLogToFile, "1") == 0;
}

static LogLevel GetLogLevelFromEnvCfg(bool controlledByCfg)
{
    const char *env = std::getenv("ASDOPS_LOG_LEVEL");
    if (controlledByCfg) {
        env = CfgCore::GetCfgCoreInstance().GetLogCfg().logLevel.c_str();
    }
    if (env == nullptr || strlen(env) > MAX_ENV_STRING_LEN) {
        return LogLevel::WARN;
    }
    std::string envLogLevel(env);
    std::transform(envLogLevel.begin(), envLogLevel.end(), envLogLevel.begin(), ::toupper);
    static std::unordered_map<std::string, LogLevel> levelMap{{"TRACE", LogLevel::TRACE}, {"DEBUG", LogLevel::DEBUG},
                                                              {"INFO", LogLevel::INFO},   {"WARN", LogLevel::WARN},
                                                              {"ERROR", LogLevel::ERROR}, {"FATAL", LogLevel::FATAL}};
    auto levelIt = levelMap.find(envLogLevel);
    return levelIt != levelMap.end() ? levelIt->second : LogLevel::WARN;
}

LogCore::LogCore()
{
    controlledByCfg_ = CfgCore::GetCfgCoreInstance().CfgFileExists();
    level_ = GetLogLevelFromEnvCfg(controlledByCfg_);
    if (GetLogToStdoutFromEnvCfg(controlledByCfg_)) {
        AddSink(std::make_shared<LogSinkStdout>());
    }
    if (GetLogToFileFromEnvCfg(controlledByCfg_)) {
        AddSink(std::make_shared<LogSinkFile>());
    }
}

LogCore &LogCore::Instance()
{
    static LogCore logCore;
    return logCore;
}

LogLevel LogCore::GetLogLevel() const { return level_; }

void LogCore::SetLogLevel(LogLevel level) { level_ = level; }

void LogCore::Log(const char *log, uint64_t logLen)
{
    for (auto &sink : sinks_) {
        sink->Log(log, logLen);
    }
}

void LogCore::DeleteLogFileSink()
{
    sinks_.pop_back();
}

void LogCore::AddSink(std::shared_ptr<LogSink> sink)
{
    if (sink == nullptr) {
        MKI_LOG(ERROR) << "sink is NULL";
        return;
    }
    sinks_.push_back(sink);
}

const std::vector<std::shared_ptr<LogSink>> &LogCore::GetAllSinks() const { return sinks_; }
} // namespace Mki
