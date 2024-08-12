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
#include "mki/utils/status/status.h"

namespace Mki {
TEST(StatusTest, Base)
{
    Mki::Status st = Mki::Status::FailStatus(3, "dd");
    std::cout << st.ToString() << std::endl;
    Mki::Status st1 = st;
    Mki::Status st2;
    st2 = st1;
}
TEST(StatusTest, Base1)
{
    Mki::Status st = Mki::Status::OkStatus();
    EXPECT_EQ(st.Ok(), true);
    EXPECT_EQ(st.ToString(), "ok");
    EXPECT_EQ(st.Code(), 0);
    EXPECT_EQ(st.Message(), "");
}

TEST(StatusTest, Base2)
{
    Mki::Status status;
    Mki::Status status_copy = status;
}

TEST(StatusTest, Code)
{
    Mki::Status st = Mki::Status::FailStatus(3, "dd");
    EXPECT_EQ(st.Code(), 3);
}
} // namespace Mki
