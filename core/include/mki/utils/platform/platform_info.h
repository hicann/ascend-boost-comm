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
#ifndef MKI_PLATFORM_INFO_H
#define MKI_PLATFORM_INFO_H

#include <string>

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

private:
    PlatformInfo();

    void Init();
    bool Inited() const;

    bool GetAicoreIntrinsic(const std::string &intrinsicName);

    bool inited_ = false;

    // platform
    PlatformType platformType_ = PlatformType::PLATFORM_INVALID;

    fe::PlatFormInfos platformInfo_;
};
} // namespace Mki

#endif
