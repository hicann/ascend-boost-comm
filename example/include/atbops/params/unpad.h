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
#ifndef ATBOPS_PARAMS_UNPAD_H
#define ATBOPS_PARAMS_UNPAD_H

#include <cstdint>
#include <string>
#include <sstream>
#include <vector>
#include "mki/utils/compare/compare.h"

namespace AtbOps {
namespace OpParam {
struct Unpad {
    bool operator==(const Unpad &other) const
    {
        (void)other;
        return true;
    }
};
} // namespace OpParam
} // namespace AtbOps

#endif // ATBOPS_PARAMS_UNPAD_H