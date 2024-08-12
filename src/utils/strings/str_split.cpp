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
#include <sstream>
#include "mki/utils/log/log.h"
#include "mki/utils/strings/str_split.h"

namespace Mki {
constexpr int64_t MAX_USER_STRING_LENGTH = 10000;

void StrSplit(const std::string &text, const char delimiter, std::vector<std::string> &result)
{
    std::istringstream iss(text);
    std::string subStr;
    uint32_t lineNum = 0;
    while (getline(iss, subStr, delimiter)) {
        if (lineNum > MAX_USER_STRING_LENGTH) {
            MKI_LOG(ERROR) << "file lineNum is out of range, lineNum : " << lineNum;
            break;
        }
        lineNum++;
        result.push_back(subStr);
    }
}
} // namespace Mki
