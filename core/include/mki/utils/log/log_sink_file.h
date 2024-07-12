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
#ifndef MKI_UTILS_LOG_FILE_H
#define MKI_UTILS_LOG_FILE_H

#include <fstream>
#include <mutex>
#include "mki/utils/log/log_sink.h"

namespace Mki {
class LogSinkFile : public LogSink {
public:
    LogSinkFile();
    ~LogSinkFile() override;
    void Log(const char *log, uint64_t logLen) override;

private:
    LogSinkFile(const LogSinkFile &) = delete;
    const LogSinkFile &operator=(const LogSinkFile &) = delete;
    void Init();
    void OpenFile();
    void DeleteOldestFile();
    std::string GetNewLogFilePath();
    bool IsDiskAvailable();
    void MakeLogDir();
    void CloseFile();
    std::string GetHomeDir();

private:
    std::string boostType_;
    std::string logDir_;
    bool isFlush_ = false;
    int currentFd_ = -1;
    uint64_t currentFileSize_ = 0;
    std::mutex mutex_;
};
} // namespace Mki
#endif
