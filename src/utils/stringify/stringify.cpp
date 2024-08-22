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
#include "mki/utils/log/log.h"
#include "mki/utils/stringify/stringify.h"

namespace Mki {
std::map<size_t, StringifyFunc> Stringify::converterMap_;

Stringify::Stringify(size_t typeHashCode, StringifyFunc func)
{
    auto it = converterMap_.find(typeHashCode);
    if (it == converterMap_.end()) {
        converterMap_[typeHashCode] = func;
    } else {
        MKI_LOG(WARN) << typeHashCode << " has been registered";
    }
}

std::string Stringify::ToString(const Any &param)
{
    auto it = converterMap_.find(param.Type().hash_code());
    if (it != converterMap_.end()) {
        return it->second(param);
    }
    return std::string(param.Type().name()) + std::string(" can not be printed");
}
} // namespace Mki
