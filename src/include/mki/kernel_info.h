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
#ifndef MKI_KERNEL_INFO_H
#define MKI_KERNEL_INFO_H

#include <cstdint>
#include "mki/utils/non_copyable/non_copyable.h"
#include "mki/utils/status/status.h"
#include "mki/utils/SVector/SVector.h"

namespace Mki {
class KernelInfo : public NonCopyable {
public:
struct ConstTensorInfo {
    uint64_t argIdx = 0;
    uint64_t size = 0;
};

struct TilingExtInfo {
    uint32_t blockDim = 0;
    uint64_t tilingId = 0;
    uint64_t constTensorOffset = 0; // const tensor offset in host tiling
    uint64_t usedSize = 0; // actual tiling size used
    uint8_t *hostTilingAddr = nullptr;
    uint64_t hostTilingSize = 0;
};

struct MemsetInfo {
    uint64_t argIdx = 0;
    uint64_t size = 0;
};

struct TensorListExtInfo {
    uint8_t *tensorListAddr = nullptr;
    uint64_t tensorListSize = 0;
};

public:
    KernelInfo() = default;
    ~KernelInfo();
    void Copy(const KernelInfo &other);
    std::string ToString() const;

public:
    void Reset();
    // Args
    Status InitArgs(uint64_t len);
    uint8_t *GetArgs() const;
    uint64_t GetArgsSize() const;

    // Hwsync
    void SetHwsyncIdx(int64_t idx);
    int64_t GetHwsyncIdx() const;

    // TilingExtInfo - BlockDim / TilingId
    void SetBlockDim(uint32_t blockDim);
    uint32_t GetBlockDim() const;
    void SetTilingId(uint64_t tilingId);
    uint64_t GetTilingId() const;

    // TilingExtInfo - TilingHost
    Status AllocTilingHost(uint64_t len);
    void SetTilingHostAddr(uint8_t *addr, uint64_t len);
    uint8_t *GetTilingHostAddr() const;
    uint64_t GetTilingSize() const;

    void SetTilingUsedSize(uint64_t usedSize);
    uint64_t GetTilingUsedSize() const;

    // TensorListExtInfo
    Status AllocTensorListHost(uint64_t len);
    void SetTensorListHostAddr(uint8_t *addr, uint64_t len);
    uint8_t *GetTensorListHostAddr() const;
    uint64_t GetTensorListSize() const;

    // TilingExtInfo - ConstTensorOffset
    void SetConstTensorOffset(uint64_t offset);
    uint64_t GetConstTensorOffset() const;

    // ConstTensor
    template <typename T_SRC, typename T_DST = T_SRC, typename T_CONT = SVector<T_SRC>>
    bool AddConstTensorData(uint64_t argIdx, const T_CONT &tensorData);

    size_t GetConstTensorCount() const;
    const ConstTensorInfo &GetConstTensorInfo(size_t id) const;
    const MiniVector<ConstTensorInfo> &GetConstTensorInfos() const;

    // LaunchWithTiling
    void SetLaunchWithTiling(bool flag);
    bool GetLaunchWithTiling() const;

    // LaunchWithTensorlist
    void SetLaunchWithTensorlist(bool flag);
    bool GetLaunchWithTensorlist() const;
    size_t GetTensorListCount() const;
    const ConstTensorInfo &GetTensorListInfo(size_t id) const;
    const MiniVector<ConstTensorInfo> &GetTensorListInfos() const;
    bool AddTensorListInfo(uint64_t argIdx, uint64_t len);

    // Scratch
    MiniVector<uint64_t> &GetScratchSizes();
    const MiniVector<uint64_t> &GetScratchSizes() const;
    uint64_t GetTotalScratchSize() const;

    // Memset
    void SetMemsetInfo(uint64_t argIdx, uint64_t size);
    const MiniVector<KernelInfo::MemsetInfo> &GetMemsetInfo() const;

private:
    void ResetArgs();
    void ResetTilingInfo();
    void ResetTensorListExtInfo();
    void ResetConstTensorInfo();
    void ResetScratchSizes();
    void ResetMemsetInfo();

private:
    uint8_t *args_ = nullptr;
    uint64_t argsSize_ = 0;
    bool initFlag_{false};  // kernel info 线程间不共享
    bool launchWithTiling_{true};
    bool launchWithTensorlist_{false};
    int64_t hwsyncIdx_ = -1; // < 0: no hwsync, >= 0: hwsync arg idx
    TilingExtInfo tilingExtInfo_;
    TensorListExtInfo tensorListExtInfo_;
    MiniVector<ConstTensorInfo> constTensorInfo_;
    MiniVector<uint64_t> scratchSizes_;
    MiniVector<MemsetInfo> memsetInfo_;
};
} // namespace Mki

#endif
