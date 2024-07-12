/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mki/utils/strings/str_checker.h"
#include "mki/utils/log/log.h"

namespace Mki {
bool CheckNameValid(const std::string &name, const size_t maxLen)
{
    // 1. check the name not empty
    if (name.empty()) {
        MKI_LOG(ERROR) << "The name is empty";
        return false;
    }

    // 2. check the length of the name
    if (name.size() > maxLen) {
        MKI_LOG(ERROR) << "The length of the name " << name.size() << " > " << maxLen;
        return false;
    }

    // 3. check the name has invalid characters
    const std::vector<char> invalidChars = { '\n', '\f', '\r', '\v', '\t', '\b', '\u007f', ' ', '$'};
    for (auto c : invalidChars) {
        if (name.find(c) != std::string::npos) {
            MKI_LOG(ERROR) << "Name contains invalid characters: " << c;
            return false;
        }
    }
    return true;
}
}
