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
#ifndef MKI_UTILS_DL_DL_H
#define MKI_UTILS_DL_DL_H
#include <string>
#include <atomic>
#include "mki/utils/noncopyable/noncopyable.h"

namespace Mki {
class Dl : public NonCopyable {
public:
    explicit Dl(const std::string &dlFilePath);
    ~Dl();
    void *GetSymbol(const std::string &symbolName) const;
    bool IsValid() const;
private:
    Dl(const Dl &) = delete;
    const Dl &operator=(const Dl &) = delete;
    std::string GetNewMemFdFileName() const;
    std::atomic_bool dlSuccess_{false};
private:
    int memFd_ = 0;
    void *dlHandle_ = nullptr;
};
} // namespace Mki
#endif
