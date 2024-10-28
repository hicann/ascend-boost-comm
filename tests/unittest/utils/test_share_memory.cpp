/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <cstdlib>
#include <gtest/gtest.h>
#include "mki/utils/share_memory/share_memory.h"

namespace Mki {
TEST(TestShareMemory, ShareMemoryName)
{
    std::string shmName = "shmName";
    setenv("ATB_SHARE_MEMORY_NAME_SUFFIX", "1", 1);
    ShareMemory shm(shmName, 80);
    EXPECT_NE(shm.GetShm(), nullptr);
    std::string fullName = shmName + "1";
    sem_t *sem = sem_open(fullName.c_str(), 0);
    EXPECT_NE(shm.GetShm(), nullptr);
    sem_close(sem);
}
}