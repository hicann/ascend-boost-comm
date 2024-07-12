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
#include "mki/kernel_info.h"

#include <securec.h>

#include "mki/utils/assert/assert.h"
#include "mki/utils/fp16/fp16_t.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/tensor_utils.h"

namespace Mki {
KernelInfo::~KernelInfo()
{
    ResetArgs();
}

void KernelInfo::Reset(bool deleteTiling)
{
    hwsyncIdx_ = -1;
    ResetArgs();
    if (deleteTiling && tilingExtInfo_.hostTilingAddr != nullptr) {
        delete[] tilingExtInfo_.hostTilingAddr;
        tilingExtInfo_.hostTilingAddr = nullptr;
        tilingExtInfo_.hostTilingSize = 0;
        tilingExtInfo_.constTensorOffset = 0;
        tilingExtInfo_.usedSize = 0;
    }
    ResetTilingInfo();
    ResetConstTensorInfo();
    ResetScratchSizes();
    ResetMemsetInfo();
}

Status KernelInfo::InitArgs(uint64_t len)
{
    constexpr uint64_t maxArgsSize = 1024 * 1024; // 1mb
    MKI_CHECK(len <= maxArgsSize, "failed to check args len " << len, return Status::FailStatus(-1));

    args_ = new (std::nothrow) uint8_t[len];
    MKI_CHECK(args_ != nullptr, "failed to new args, len " << len, return Status::FailStatus(-1));
    (void)memset_s(args_, len, 0, len);

    argsSize_ = len;
    MKI_LOG(INFO) << "args inited successfully, len " << len;

    return Status::OkStatus();
}

uint8_t *KernelInfo::GetArgs() const
{
    return args_;
}

size_t KernelInfo::GetArgsSize() const
{
    return argsSize_;
}

void KernelInfo::SetHwsyncIdx(int64_t idx)
{
    hwsyncIdx_ = idx;
}

int64_t KernelInfo::GetHwsyncIdx() const
{
    return hwsyncIdx_;
}

void KernelInfo::SetBlockDim(uint32_t blockDim)
{
    constexpr uint32_t maxBlockDim = 65536;
    MKI_CHECK(blockDim < maxBlockDim, "failed to set block dim " << blockDim, return);
    tilingExtInfo_.blockDim = blockDim;
}

uint32_t KernelInfo::GetBlockDim() const
{
    return tilingExtInfo_.blockDim;
}

void KernelInfo::SetTilingId(uint64_t tilingId)
{
    tilingExtInfo_.tilingId = tilingId;
}

uint64_t KernelInfo::GetTilingId() const
{
    return tilingExtInfo_.tilingId;
}

void KernelInfo::SetTilingUsedSize(uint64_t usedSize)
{
    MKI_CHECK(usedSize <= tilingExtInfo_.hostTilingSize,
                 "failed to set tiling usedsize " << usedSize, return);
    tilingExtInfo_.usedSize = usedSize;
}

uint64_t KernelInfo::GetTilingUsedSize() const
{
    return tilingExtInfo_.usedSize;
}

Status KernelInfo::AllocTilingHost(uint64_t len)
{
    constexpr uint64_t maxTilingSize = 1024 * 1024; // 1mb
    MKI_CHECK(len > 0 && len <= maxTilingSize,
        "failed to check tiling len " << len, return Status::FailStatus(-1));
    MKI_CHECK(tilingExtInfo_.hostTilingAddr == nullptr,
        "Tiling is already alloced", return Status::FailStatus(-1));

    tilingExtInfo_.hostTilingAddr = new (std::nothrow) uint8_t[len];
    MKI_CHECK(tilingExtInfo_.hostTilingAddr != nullptr,
        "failed to new tiling, len " << len, return Status::FailStatus(-1));
    (void)memset_s(tilingExtInfo_.hostTilingAddr, len, 0, len);
    MKI_LOG(INFO) << "alloc " << len << " bytes tiling host";
    tilingExtInfo_.hostTilingSize = len;
    tilingExtInfo_.usedSize = len;
    tilingExtInfo_.constTensorOffset = len; // no const tensor

    return Status::OkStatus();
}

void KernelInfo::SetTilingHostAddr(uint8_t *addr, uint64_t len)
{
    MKI_CHECK(addr != nullptr,
        "failed to set host tiling addr to nullptr ", return);
    tilingExtInfo_.hostTilingAddr = addr;
    tilingExtInfo_.hostTilingSize = len;
    tilingExtInfo_.usedSize = len;
    tilingExtInfo_.constTensorOffset = len; // no const tensor
}

uint8_t *KernelInfo::GetTilingHostAddr() const
{
    return tilingExtInfo_.hostTilingAddr;
}

uint64_t KernelInfo::GetTilingSize() const
{
    return tilingExtInfo_.hostTilingSize;
}

void KernelInfo::SetConstTensorOffset(uint64_t offset)
{
    MKI_CHECK(offset > 0 && offset < tilingExtInfo_.hostTilingSize,
        "failed to check tiling len " << offset, return);
    MKI_LOG(INFO) << "set const tensor offset " << offset;
    tilingExtInfo_.constTensorOffset = offset;

    if (tilingExtInfo_.usedSize > offset) {
        MKI_LOG(INFO) << "fix tiling used size from " << tilingExtInfo_.usedSize << " to " << offset;
        tilingExtInfo_.usedSize = offset;
    }
}

void KernelInfo::SetMemsetInfo(uint64_t argIdx, uint64_t size)
{
    memsetInfo_.push_back({argIdx, size});
}

MiniVector<KernelInfo::MemsetInfo> &KernelInfo::GetMemsetInfo()
{
    return memsetInfo_;
}

uint64_t KernelInfo::GetConstTensorOffset() const
{
    return tilingExtInfo_.constTensorOffset;
}

template <typename T_SRC, typename T_DST = T_SRC, typename T_CONT = SVector<T_SRC>>
bool KernelInfo::AddConstTensorData(uint64_t argIdx, const T_CONT &tensorData)
{
    uint64_t offset = tilingExtInfo_.constTensorOffset;
    for (const auto &info : constTensorInfo_) {
        offset += info.size;
    }
    MKI_CHECK(offset < tilingExtInfo_.hostTilingSize,
        "failed to check const tensor offset " << offset, return false);

    uint64_t len = Utils::GetConstTensorSize<T_SRC, T_DST, T_CONT>(tensorData);

    MKI_LOG(INFO) << "add const tensor info " << constTensorInfo_.size()
                  << ", argIdx " << argIdx << ", offset " << offset << ", len " << len;

    if (std::is_same<T_SRC, T_DST>::value) {
        MKI_LOG(INFO) << "copy const tensor without transfer";
        auto ret = memcpy_s(tilingExtInfo_.hostTilingAddr + offset, tilingExtInfo_.hostTilingSize - offset,
                            static_cast<const void *>(tensorData.data()), tensorData.size() * sizeof(T_SRC));
        MKI_CHECK(ret == EOK, "failed to copy const tensor data", return false);
    } else {
        MKI_LOG(INFO) << "copy const tensor with transfer";
        T_DST tensorDataTransfer[tensorData.size()];
        for (size_t i = 0; i < tensorData.size(); i++) {
            tensorDataTransfer[i] = static_cast<T_DST>(tensorData[i]);
        }
        auto ret = memcpy_s(tilingExtInfo_.hostTilingAddr + offset, tilingExtInfo_.hostTilingSize - offset,
                            tensorDataTransfer, tensorData.size() * sizeof(T_DST));
        MKI_CHECK(ret == EOK, "failed to copy const tensor data", return false);
    }
    constTensorInfo_.push_back({argIdx, len});

    return true;
}

size_t KernelInfo::GetConstTensorCount() const
{
    return constTensorInfo_.size();
}

const KernelInfo::ConstTensorInfo &KernelInfo::GetConstTensorInfo(size_t id) const
{
    // id out of bound will throw
    return constTensorInfo_.at(id);
}

MiniVector<uint64_t> &KernelInfo::GetScratchSizes()
{
    return scratchSizes_;
}

uint64_t KernelInfo::GetTotalScratchSize() const
{
    uint64_t sum = 0;
    for (auto wkspSize : scratchSizes_) {
        sum += wkspSize;
    }
    return sum;
}

std::string KernelInfo::ToString() const
{
    std::stringstream ss;

    ss << "hwsyncIdx: " << hwsyncIdx_ << ", tiling ext info: blockDim " << tilingExtInfo_.blockDim
       << ", tilingId " << tilingExtInfo_.tilingId << ", constTensorOffset " << tilingExtInfo_.constTensorOffset
       << ", usedSize " << tilingExtInfo_.usedSize << ", tilingSize " << tilingExtInfo_.hostTilingSize;
    ss << ", constTensorInfo: ";
    for (size_t i = 0; i < constTensorInfo_.size(); i++) {
        ss << ", constTensor" << i << ": argIdx " << constTensorInfo_[i].argIdx
           << ", size " << constTensorInfo_[i].size;
    }
    ss << ", scratch sizes: " << scratchSizes_;
    for (size_t i = 0; i < memsetInfo_.size(); i++) {
        ss << ", memset" << i << ": argIdx " << memsetInfo_[i].argIdx
           << ", size " << memsetInfo_[i].size;
    }

    return ss.str();
}

void KernelInfo::Copy(const KernelInfo &kernelInfo)
{
    hwsyncIdx_ = kernelInfo.hwsyncIdx_;
    tilingExtInfo_.blockDim = kernelInfo.tilingExtInfo_.blockDim;
    tilingExtInfo_.tilingId = kernelInfo.tilingExtInfo_.tilingId;
    tilingExtInfo_.constTensorOffset = kernelInfo.tilingExtInfo_.constTensorOffset;
    tilingExtInfo_.usedSize = kernelInfo.tilingExtInfo_.usedSize;
    constTensorInfo_ = kernelInfo.constTensorInfo_;
    scratchSizes_ = kernelInfo.scratchSizes_;
    memsetInfo_ = kernelInfo.memsetInfo_;
}

void KernelInfo::ResetArgs()
{
    if (args_ != nullptr) {
        delete[] args_;
        args_ = nullptr;
    }
    argsSize_ = 0;
}

void KernelInfo::ResetTilingInfo()
{
    // No checking return value is ok
    tilingExtInfo_.blockDim = 0;
    tilingExtInfo_.tilingId = 0;
}

void KernelInfo::ResetConstTensorInfo()
{
    constTensorInfo_.clear();
}

void KernelInfo::ResetScratchSizes()
{
    scratchSizes_.clear();
}

void KernelInfo::ResetMemsetInfo()
{
    memsetInfo_.clear();
}

template bool KernelInfo::AddConstTensorData<int32_t>(uint64_t, const SVector<int32_t> &);
template bool KernelInfo::AddConstTensorData<int64_t>(uint64_t, const SVector<int64_t> &);
template bool KernelInfo::AddConstTensorData<fp16_t>(uint64_t, const SVector<fp16_t> &);
template bool KernelInfo::AddConstTensorData<float>(uint64_t, const SVector<float> &);
template bool KernelInfo::AddConstTensorData<float, int32_t>(uint64_t, const SVector<float> &);
template bool KernelInfo::AddConstTensorData<float, fp16_t>(uint64_t, const SVector<float> &);
template bool KernelInfo::AddConstTensorData<int32_t>(uint64_t, const std::vector<int32_t> &);
template bool KernelInfo::AddConstTensorData<int8_t>(uint64_t, const std::vector<int8_t> &);
} // namespace Mki
