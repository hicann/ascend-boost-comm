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
#ifndef MKI_UTILS_LOG_LOG_SINK_STDOUT_H
#define MKI_UTILS_LOG_LOG_SINK_STDOUT_H
#include <mutex>
#include "mki/utils/log/log_sink.h"

namespace Mki {
class LogSinkStdout : public LogSink {
public:
    LogSinkStdout() = default;
    ~LogSinkStdout() override = default;
    void Log(const char *log, uint64_t logLen) override;

private:
    std::mutex mtx_;
};
} // namespace Mki
#endif
