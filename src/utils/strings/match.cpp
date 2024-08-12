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
#include "mki/utils/strings/match.h"
#include <cstring>

namespace Mki {
bool StartsWith(const std::string &text, const std::string &prefix)
{
    return prefix.empty() || (text.size() >= prefix.size() && memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

bool EndsWith(const std::string &text, const std::string &suffix)
{
    return suffix.empty() || (text.size() >= suffix.size() &&
                              memcmp(text.data() + (text.size() - suffix.size()), suffix.data(), suffix.size()) == 0);
}
} // namespace Mki
