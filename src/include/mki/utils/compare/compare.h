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
#ifndef MKI_UTILS_COMPARE_COMPARE_H
#define MKI_UTILS_COMPARE_COMPARE_H
#include <cmath>

namespace Mki {
namespace Utils {
template<class T>
class Compare {
public:
    static bool IsEqual(const T &lh, const T &rh)
    {
        return lh == rh;
    }
};

template<>
class Compare<float> {
public:
    static bool IsEqual(const float &lh, const float &rh)
    {
        return std::abs(lh - rh) < 0.000001; // float类型精度0.000001
    }
};

template<>
class Compare<double> {
public:
    static bool IsEqual(const double &lh, const double &rh)
    {
        return std::abs(lh - rh) < 0.000000001; // double类型精度0.000000001
    }
};

}
} // namespace Mki
#endif
