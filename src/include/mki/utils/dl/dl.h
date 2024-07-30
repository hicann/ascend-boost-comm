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
#ifndef MKI_UTILS_DL_DL_H
#define MKI_UTILS_DL_DL_H
#include <string>
#include <atomic>
#include "mki/utils/non_copyable/non_copyable.h"

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
