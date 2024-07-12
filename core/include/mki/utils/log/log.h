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
#ifndef MKI_UTILS_LOG_LOG_H
#define MKI_UTILS_LOG_LOG_H
#include "mki/utils/log/log_stream.h"
#include "mki/utils/log/log_core.h"
#include "mki/utils/log/log_sink.h"
#include "mki/utils/log/log_entity.h"

#define MKI_LOG(level) MKI_LOG_##level

#define MKI_FLOG(level, format, ...) MKI_FLOG_##level(format, __VA_ARGS__)

#define MKI_LOG_IF(condition, level)                                                                                   \
    if (condition)                                                                                                     \
    MKI_LOG(level)

#define MKI_LOG_TRACE                                                                                                  \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE)
#define MKI_LOG_DEBUG                                                                                                  \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG)
#define MKI_LOG_INFO                                                                                                   \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel())                                           \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO)
#define MKI_LOG_WARN                                                                                                   \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel())                                           \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN)
#define MKI_LOG_ERROR                                                                                                  \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR)
#define MKI_LOG_FATAL                                                                                                  \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL)

#define MKI_FLOG_TRACE(format, ...)                                                                                    \
    if (Mki::LogLevel::TRACE >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::TRACE).Format(format, __VA_ARGS__)
#define MKI_FLOG_DEBUG(format, ...)                                                                                    \
    if (Mki::LogLevel::DEBUG >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::DEBUG).Format(format, __VA_ARGS__)
#define MKI_FLOG_INFO(format, ...)                                                                                     \
    if (Mki::LogLevel::INFO >= Mki::LogCore::Instance().GetLogLevel())                                           \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::INFO).Format(format, __VA_ARGS__)
#define MKI_FLOG_WARN(format, ...)                                                                                     \
    if (Mki::LogLevel::WARN >= Mki::LogCore::Instance().GetLogLevel())                                           \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::WARN).Format(format, __VA_ARGS__)
#define MKI_FLOG_ERROR(format, ...)                                                                                    \
    if (Mki::LogLevel::ERROR >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::ERROR).Format(format, __VA_ARGS__)
#define MKI_FLOG_FATAL(format, ...)                                                                                    \
    if (Mki::LogLevel::FATAL >= Mki::LogCore::Instance().GetLogLevel())                                          \
    Mki::LogStream(__FILE__, __LINE__, __FUNCTION__, Mki::LogLevel::FATAL).Format(format, __VA_ARGS__)

#endif
