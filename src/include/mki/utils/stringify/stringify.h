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
#ifndef MKI_STRINGIFY_H
#define MKI_STRINGIFY_H

#include <functional>
#include <map>
#include <string>
#include "mki/utils/any/any.h"

namespace Mki {
using StringifyFunc = std::function<std::string(const Any &)>;

class Stringify {
public:
    Stringify(size_t typeHashCode, StringifyFunc func);
    static std::string ToString(const Any &param);
private:
    static std::map<size_t, StringifyFunc> converterMap_;
};
} // namespace Mki

#define REG_STRINGIFY(typename, func) \
    static Mki::Stringify g_register_##func(typeid(typename).hash_code(), func)

#endif
