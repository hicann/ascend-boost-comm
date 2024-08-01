/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include <gtest/gtest.h>
#include "mki/utils/rt/backend/rt_backend.h"

TEST(RtBackendTest, ModuleCreate) {
    Mki::RtBackend rtBackend;
    MkiRtModuleInfo moduleInfo;
    MkiRtModule module;
    moduleInfo.type = MKIRT_MODULE_OBJECT;
    moduleInfo.version = 0;
    moduleInfo.data = 0;
    moduleInfo.dataLen = 0;
    moduleInfo.magic = 0x41494343U;
    ASSERT_EQ(rtBackend.ModuleCreate(nullptr, &module), MKIRT_ERROR_NOT_INITIALIZED);
    ASSERT_EQ(rtBackend.ModuleCreate(&moduleInfo, nullptr), MKIRT_ERROR_NOT_INITIALIZED);
}
