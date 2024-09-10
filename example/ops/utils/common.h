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
#ifndef ASCEND_OPS_COMMON_H
#define ASCEND_OPS_COMMON_H

#include <mki/utils/SVector/SVector.h>

namespace AtbOps {
template<class T>
bool CheckParamAllPositive(SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] <= 0) {
            return false;
        }
    }
    return true;
}
template<class T>
bool CheckParamAnyNegative(SVector<T> &param)
{
    for (size_t i = 0; i < param.size(); i++) {
        if (param[i] < 0) {
            return true;
        }
    }
    return false;
}
} // namespace AtbOps

#endif