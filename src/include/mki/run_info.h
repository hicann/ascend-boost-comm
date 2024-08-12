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
#ifndef MKI_RUN_INFO_H
#define MKI_RUN_INFO_H

#include <cstdint>
#include <string>
#include "mki/utils/non_copyable/non_copyable.h"

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

    void Copy(const RunInfo &runInfo);

private:
    // used by User
    void *stream_ = nullptr;
    uint8_t *scratchAddr_ = nullptr;
    uint8_t *tilingDeviceAddr_ = nullptr;
};
} // namespace Mki

#endif
