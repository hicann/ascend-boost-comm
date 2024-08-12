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
#ifndef MKI_UTILS_TIME_TIMER_H
#define MKI_UTILS_TIME_TIMER_H
#include <cstdint>

namespace Mki {
class Timer {
public:
    Timer();
    ~Timer();
    uint64_t ElapsedMicroSecond();
    void Reset();

private:
    uint64_t GetCurrentTimepoint() const;

private:
    uint64_t startTimepoint_ = 0;
};
} // namespace Mki
#endif
