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
#ifndef MKI_UTILS_PLATFORM_PLATFORM_INFO_H
#define MKI_UTILS_PLATFORM_PLATFORM_INFO_H

#include <string>
#include <unordered_map>
#include "mki/utils/platform/platform_manager.h"

namespace Mki {
enum class PlatformType {
    ASCEND_310P = 0,
    ASCEND_910A,
    ASCEND_910B,
    PLATFORM_INVALID,
};

enum class CoreType : int {
    CORE_TYPE_VECTOR = 0,
    CORE_TYPE_CUBE = 1,
};

class PlatformInfo {
public:
    static PlatformInfo &Instance();

    uint32_t GetCoreNum(CoreType type);

    uint64_t GetL2Size();
    uint64_t GetL1Size();
    uint64_t GetL0ASize();
    uint64_t GetL0BSize();
    uint64_t GetL0CSize();
    uint64_t GetUbSize();

    bool SupportL0c2out();

    PlatformType GetPlatformType() const;
    std::string GetPlatformName() const;

private:
    PlatformInfo();

    void Init();
    bool Inited() const;

    bool GetAicoreIntrinsic(const std::string &intrinsicName);

    bool inited_ = false;

    // platform
    PlatformType platformType_ = PlatformType::PLATFORM_INVALID;
    std::string platformName_;

    fe::PlatFormInfos platformInfo_;
};
} // namespace Mki

#endif
