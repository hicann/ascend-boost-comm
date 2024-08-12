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
#ifndef MKI_UTILS_ASSERT_ASSERT_H
#define MKI_UTILS_ASSERT_ASSERT_H

#include "mki/utils/log/log.h"

namespace Mki {
namespace Utils {
#define MKI_CHECK(condition, logExpr, handleExpr)                                                                   \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        MKI_LOG(ERROR) << logExpr;                                                                                  \
        handleExpr;                                                                                                 \
    }                                                                                                               \
} while (0)

#define MKI_CHECK_NO_LOG(condition, handleExpr)                                                                     \
do {                                                                                                                \
    if (!(condition)) {                                                                                             \
        handleExpr;                                                                                                 \
    }                                                                                                               \
} while (0)
}

#define UNUSED_VALUE(x) (void)(x)
} // namespace Mki
#endif
