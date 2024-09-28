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

#ifndef MKI_ENV_ENV_H
#define MKI_ENV_ENV_H

#include <cstdlib>
#include <cstring>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

namespace Mki {
constexpr size_t MAX_ENV_STRING_LEN = 12800;

inline const char *GetEnv(const char *name)
{
    MKI_CHECK(name != nullptr, "env name is nullptr!", return nullptr);
    const char *env = std::getenv(name);
    if (env != nullptr && strlen(env) <= MAX_ENV_STRING_LEN) {
        return env;
    }
    MKI_LOG(WARN) << "env " << name << " is too long or not exist!";
    return nullptr;
}
} // namespace mki

#endif // MKI_ENV_ENV_H