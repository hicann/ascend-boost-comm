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
#include <thread>
#include "mki/utils/time/timer.h"

namespace {
TEST(TimerTest, ElapsedMicroSecond)
{
    Mki::Timer timer;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << timer.ElapsedMicroSecond() << std::endl;

    timer.Reset();
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << timer.ElapsedMicroSecond() << std::endl;

    timer.Reset();
    {
        std::vector<int> data(10);
    }
    std::cout << timer.ElapsedMicroSecond() << std::endl;

    timer.Reset();
    std::cout << timer.ElapsedMicroSecond() << std::endl;
}
} // namespace
