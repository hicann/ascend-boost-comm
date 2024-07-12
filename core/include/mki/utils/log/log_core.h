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
#ifndef MKI_UTILS_LOG_LOGCORE_H
#define MKI_UTILS_LOG_LOGCORE_H
#include <memory>
#include <vector>
#include "mki/utils/log/log_entity.h"
#include "mki/utils/log/log_sink.h"
#include "mki/utils/svector/svector.h"

namespace Mki {
class LogCore {
public:
    LogCore();
    virtual ~LogCore() = default;
    static LogCore &Instance();
    LogLevel GetLogLevel() const;
    void SetLogLevel(LogLevel level);
    void Log(const char *log, uint64_t logLen);
    void DeleteLogFileSink();
    void AddSink(std::shared_ptr<LogSink> sink);
    const std::vector<std::shared_ptr<LogSink>> &GetAllSinks() const;

private:
    std::vector<std::shared_ptr<LogSink>> sinks_;
    LogLevel level_ = LogLevel::INFO;
};
} // namespace Mki
#endif
