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

namespace Mki {
static bool GetLogToStdoutFromEnv()
{
    const char *envLogToStdout = std::getenv("MKI_LOG_TO_STDOUT");
    return envLogToStdout != nullptr && strcmp(envLogToStdout, "1") == 0;
}

static bool GetLogToFileFromEnv()
{
    const char *envLogToFile = std::getenv("MKI_LOG_TO_FILE");
    return envLogToFile != nullptr && strcmp(envLogToFile, "1") == 0;
}

static LogLevel GetLogLevelFromEnv()
{
    const char *env = std::getenv("MKI_LOG_LEVEL");
    if (env == nullptr) {
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
