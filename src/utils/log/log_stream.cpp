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
#include <securec.h>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <cstdarg>
#include <unistd.h>
#include <syscall.h>
#include "mki/utils/log/log_core.h"
#include "mki/utils/log/log_stream.h"

namespace Mki {
thread_local std::ostringstream g_stream;

thread_local long g_threadId = -1;
long GetThreadId()
{
    if (g_threadId == -1) {
        g_threadId = syscall(SYS_gettid);
    }
    return g_threadId;
}

LogStream::LogStream(const char *filePath, int line, const char *funcName, LogLevel level) : stream_(g_stream)
{
    if (filePath != nullptr && funcName != nullptr) {
        const char *str = strrchr(filePath, '/');
        logEntity_.fileName = str ? str + 1 : filePath;
        logEntity_.time = std::chrono::system_clock::now();
        logEntity_.level = level;
        logEntity_.threadId = GetThreadId();
        logEntity_.funcName = funcName;
        logEntity_.line = line;
    }

    constexpr int microSecond = 1000000;
    int us = std::chrono::duration_cast<std::chrono::microseconds>(
        logEntity_.time.time_since_epoch()).count() % microSecond;
    std::time_t tmpTime = std::chrono::system_clock::to_time_t(logEntity_.time);
    struct tm timeinfo;
    localtime_r(&tmpTime, &timeinfo);
    char timeStr[64] = {0}; // 64 日志buffer长度
    (void)strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);

    stream_.str("");
    stream_ << "[" << timeStr << "." << us << "] [" << LogLevelToString(logEntity_.level) << "] ["
            << logEntity_.threadId << "] [" << logEntity_.fileName << ":" << logEntity_.line << "] ";
}

void LogStream::Format(const char *format, ...)
{
    useStream_ = false;
    std::string logHead = stream_.str();
    const int logContentSize = 1024; // 1024 日志内容最大支持1024
    const int logBufferSize = logHead.size() + logContentSize;
    char logBuffer[logBufferSize + 1] = {0};

    int ret = memcpy_s(logBuffer, logBufferSize, logHead.c_str(), logHead.size());
    if (ret != 0) {
        std::cout << "mki_log memcpy_s fail, error:" << ret << std::endl;
        return;
    }

    va_list args;
    va_start(args, format);
    int logLen = vsnprintf_s(logBuffer + logHead.size(), logContentSize + 1, logContentSize, format, args);
    va_end(args);
    if (logLen < 0) {
        std::cout << "mki_log vsnprintf_s fail" << std::endl;
        return;
    }

    logBuffer[logHead.size() + logLen] = '\n';
    LogCore::Instance().Log(logBuffer, logHead.size() + logLen + 1);
}

LogStream::~LogStream()
{
    if (useStream_) {
        stream_ << "\n";
        std::string log = stream_.str();
        LogCore::Instance().Log(log.c_str(), log.size());
    }
}
} // namespace Mki
