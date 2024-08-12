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

void RunInfo::Copy(const RunInfo &runInfo)
{
    stream_ = runInfo.stream_;
    scratchAddr_ = runInfo.scratchAddr_;
    tilingDeviceAddr_ = runInfo.tilingDeviceAddr_;
}
} // namespace Mki
