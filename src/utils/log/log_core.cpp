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
#include "mki/utils/log/log_core.h"
#include <cstdlib>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include "mki/utils/log/log_sink_stdout.h"
#include "mki/utils/log/log_sink_file.h"
#include "mki/utils/log/log.h"
#include "mki/utils/env/env.h"

namespace Mki {
static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("ASDOPS_LOG_TO_STDOUT");
    return envLogToStdout != nullptr && strlen(envLogToStdout) <= MAX_ENV_STRING_LEN &&
           strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnv()
{
    const char *envLogToFile = std::getenv("ASDOPS_LOG_TO_FILE");
    return envLogToFile != nullptr && strlen(envLogToFile) <= MAX_ENV_STRING_LEN && strcmp(envLogToFile, "1") == 0;
}

static LogLevel GetLogLevelFromEnv()
{
    const char *env = std::getenv("ASDOPS_LOG_LEVEL");
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
    level_ = GetLogLevelFromEnv();
    if (GetLogToStdoutFromEnv()) {
        AddSink(std::make_shared<LogSinkStdout>());
    }
    if (GetLogToFileFromEnv()) {
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
