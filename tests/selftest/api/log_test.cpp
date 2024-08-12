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
#include <gtest/gtest.h>
#include <cstdlib>
#include "mki/utils/SVector/SVector.h"
#include "mki/utils/log/log.h"
#include "mki/utils/log/log_entity.h"
#include "mki/utils/log/log_sink_file.h"
#include "mki/utils/log/log_sink_stdout.h"
#include "mki/utils/log/log_core.h"

namespace {
TEST(LogCore, InitTest1)
{
    char envLogToStdout[] = "MKI_LOG_TO_STDOUT=0";
    char envLogLevel[] = "MKI_LOG_LEVEL=WARN";
    putenv(envLogToStdout);
    putenv(envLogLevel);
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    std::vector<std::shared_ptr<Mki::LogSink>> sinks = logCore.GetAllSinks();
    EXPECT_EQ(logLevel, Mki::LogLevel::WARN);
    EXPECT_EQ(sinks.size(), 0);
}

TEST(LogCore, InitTest2)
{
    char envLogToStdout[] = "MKI_LOG_TO_STDOUT=1";
    char envLogLevel[] = "MKI_LOG_LEVEL=ErRoR";
    putenv(envLogToStdout);
    putenv(envLogLevel);
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    std::vector<std::shared_ptr<Mki::LogSink>> sinks = logCore.GetAllSinks();
    EXPECT_EQ(logLevel, Mki::LogLevel::ERROR);
    EXPECT_EQ(sinks.size(), 1);
    MKI_LOG(INFO) << "dd";
    MKI_LOG(WARN) << "dd";
}

TEST(LogCore, InitTest3)
{
    char envLogToStdout[] = "MKI_LOG_TO_STDOUT=avc";
    char envLogLevel[] = "MKI_LOG_LEVEL=ERROT";
    putenv(envLogToStdout);
    putenv(envLogLevel);
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    std::vector<std::shared_ptr<Mki::LogSink>> sinks = logCore.GetAllSinks();
    EXPECT_EQ(logLevel, Mki::LogLevel::WARN);
    EXPECT_EQ(sinks.size(), 0);
}

TEST(LogCore, InitTest4)
{
    unsetenv("MKI_LOG_TO_STDOUT");
    unsetenv("MKI_LOG_LEVEL");
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    std::vector<std::shared_ptr<Mki::LogSink>> sinks = logCore.GetAllSinks();
    EXPECT_EQ(logLevel, Mki::LogLevel::WARN);
    EXPECT_EQ(sinks.size(), 0);
}

TEST(LogCore, LogTest1)
{
    char envLogToStdout[] = "MKI_LOG_TO_STDOUT=1";
    char envLogLevel[] = "MKI_LOG_LEVEL=INFO";
    putenv(envLogToStdout);
    putenv(envLogLevel);
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    EXPECT_EQ(logLevel, Mki::LogLevel::INFO);
    constexpr int a = 111;
    constexpr int b = 222;
    MKI_FLOG(INFO, "value of number a is %d,value of number b is %d\n", a, b);
    MKI_FLOG(DEBUG, "value of number b is %d,value of number a is %d\n", b, a);
}

TEST(LogCore, LogTest2)
{
    char envLogToStdout[] = "MKI_LOG_TO_STDOUT=1";
    char envLogToFile[] = "MKI_LOG_TO_FILE=1";
    char envLogLevel[] = "MKI_LOG_LEVEL=INFO";
    putenv(envLogToStdout);
    putenv(envLogLevel);
    putenv(envLogToFile);
    Mki::LogCore logCore;
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    EXPECT_EQ(logLevel, Mki::LogLevel::INFO);
    constexpr int a = 111;
    constexpr int b = 222;
    MKI_FLOG(INFO, "value of number a is %d,value of number b is %d", a, b);
    MKI_FLOG(DEBUG, "value of number b is %d,value of number a is %d", b, a);
    MKI_LOG(INFO) << "AAAAAAAAA";
}

TEST(LogCore, LogTest3)
{
    Mki::LogCore logCore;
    logCore.SetLogLevel(Mki::LogLevel::DEBUG);
    Mki::LogLevel logLevel = logCore.GetLogLevel();
    EXPECT_EQ(logLevel, Mki::LogLevel::DEBUG);
}

TEST(LogEntity, Trace) {
    EXPECT_EQ(std::string("trace"), Mki::LogLevelToString(Mki::LogLevel::TRACE));
}

TEST(LogEntity, Debug) {
    EXPECT_EQ(std::string("debug"), Mki::LogLevelToString(Mki::LogLevel::DEBUG));
}

TEST(LogEntity, Info) {
    EXPECT_EQ(std::string("info"), Mki::LogLevelToString(Mki::LogLevel::INFO));
}

TEST(LogEntity, Warn) {
    EXPECT_EQ(std::string("warn"), Mki::LogLevelToString(Mki::LogLevel::WARN));
}

TEST(LogEntity, Error) {
    EXPECT_EQ(std::string("error"), Mki::LogLevelToString(Mki::LogLevel::ERROR));
}

TEST(LogEntity, Fatal) {
    EXPECT_EQ(std::string("fatal"), Mki::LogLevelToString(Mki::LogLevel::FATAL));
}

TEST(LogEntity, Unknown) {
    EXPECT_EQ(std::string("unknown"), Mki::LogLevelToString(static_cast<Mki::LogLevel>(100)));
}

TEST(LogEntity, AddSink) {
    Mki::LogCore logCore;
    logCore.AddSink(std::make_shared<Mki::LogSinkStdout>());
}
} // namespace
