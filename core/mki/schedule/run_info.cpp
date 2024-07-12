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
#include "mki/run_info.h"

#include "securec.h"

#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

namespace Mki {
RunInfo::~RunInfo() {}

void RunInfo::Reset()
{
    scratchAddr_ = nullptr;
    tilingDeviceAddr_ = nullptr;
}

void RunInfo::SetStream(void *stream)
{
    MKI_CHECK(stream != nullptr, "stream is nullptr", return);
    stream_ = stream;
}

void *RunInfo::GetStream() const { return stream_; }

void RunInfo::SetScratchDeviceAddr(uint8_t *addr)
{
    MKI_CHECK(addr != nullptr, "workspace device addr is nullptr", return);
    scratchAddr_ = addr;
}

uint8_t *RunInfo::GetScratchDeviceAddr() const { return scratchAddr_; }

void RunInfo::SetTilingDeviceAddr(uint8_t *addr)
{
    MKI_CHECK(addr != nullptr, "tiling device addr is nullptr", return);
    tilingDeviceAddr_ = addr;
}

uint8_t *RunInfo::GetTilingDeviceAddr() const { return tilingDeviceAddr_; }

std::string RunInfo::ToString() const
{
    std::stringstream ss;

#ifdef _DEBUG
    ss << "stream: " << stream_;
    ss << ", workspaceAddr: " << (void *)scratchAddr_ << ", tilingDeviceAddr: " << (void *)tilingDeviceAddr_;
#else
    ss << "stream: " << (stream_ != nullptr) << ", workspaceAddr: " << (scratchAddr_ != nullptr)
       << ", tilingDeviceAddr: " << (tilingDeviceAddr_ != nullptr);
#endif
    ss << std::endl;

    return ss.str();
}

void RunInfo::CopyTo(RunInfo &runInfo) const
{
    runInfo.Reset();
    runInfo.stream_ = stream_;
    runInfo.scratchAddr_ = scratchAddr_;
    runInfo.tilingDeviceAddr_ = tilingDeviceAddr_;
}
} // namespace Mki
