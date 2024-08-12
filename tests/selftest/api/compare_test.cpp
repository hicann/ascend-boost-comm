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
#include "mki/utils/compare/compare.h"

namespace {
TEST(CompareTest, IsEqual) {
    int a = 5;
    int b = 5;
    EXPECT_TRUE(Mki::Utils::Compare<int>::IsEqual(a, b));
    double c = 3.14;
    double d = 3.14;
    EXPECT_TRUE(Mki::Utils::Compare<double>::IsEqual(c, d));
    std::string e = "hello";
    std::string f = "world";
    EXPECT_FALSE(Mki::Utils::Compare<std::string>::IsEqual(e, f));
}
} // namespace
