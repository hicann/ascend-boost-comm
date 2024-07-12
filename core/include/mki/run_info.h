/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
#ifndef MKI_RUNINFO_H
#define MKI_RUNINFO_H

#include <cstdint>
#include <string>

#include "mki/utils/noncopyable/noncopyable.h"

namespace Mki {
class RunInfo : public NonCopyable {
public:
    RunInfo() = default;
    ~RunInfo();

public:
    void Reset();

    void SetStream(void *stream);
    void *GetStream() const;

    void SetScratchDeviceAddr(uint8_t *addr);
    uint8_t *GetScratchDeviceAddr() const;

    void SetTilingDeviceAddr(uint8_t *addr);
    uint8_t *GetTilingDeviceAddr() const;

    std::string ToString() const;

    void CopyTo(RunInfo &runInfo) const;

private:
    // used by User
    void *stream_ = nullptr;
    uint8_t *scratchAddr_ = nullptr;
    uint8_t *tilingDeviceAddr_ = nullptr;
};
} // namespace Mki

#endif
