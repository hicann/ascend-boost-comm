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
#include "mki/utils/memset/memset_launcher.h"

#include <securec.h>

#include "mki/base/kernel_base.h"
#include "mki/base/op_register.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "mki/utils/rt/rt.h"

namespace Mki {
static constexpr int32_t MEMSET_MAX_TENSOR_NUM = 8;
static constexpr size_t BLOCK_BYTES = 32;
static constexpr size_t BLOCK_BYTES_ONCE = 32 * 8; // handle 256bytes at once

class MemsetKernel : public KernelBase {
struct MemsetTilingData {
    uint32_t size[MEMSET_MAX_TENSOR_NUM] = {0};
    uint32_t sizeBlock[MEMSET_MAX_TENSOR_NUM] = {0};
    uint32_t sizeLoop[MEMSET_MAX_TENSOR_NUM] = {0};
    uint32_t maxUb[MEMSET_MAX_TENSOR_NUM] = {0}; // align to block
};

struct MemsetArgs {
    void *tensors[MEMSET_MAX_TENSOR_NUM];
    void *tiling;
    MemsetTilingData tilingData;
};

public:
    explicit MemsetKernel(const std::string &opName, KernelHandle handle) noexcept : KernelBase(opName, handle)
    {
        launchBufferSize_ = sizeof(MemsetTilingData);
    }

    bool CanSupport(const LaunchParam &launchParam) const override
    {
        return false;
    }

    Kernel *Clone() const override
    {
        auto kernel = new MemsetKernel(this->GetName(), this->GetHandle());
        kernel->Copy(*this);
        return kernel;
    }

    Status InitImpl(const LaunchParam &launchParam) override
    {
        return Status::FailStatus(1);
    }

    uint32_t MemsetTiling(MiniVector<KernelInfo::MemsetInfo> &memsetInfo, MemsetTilingData &tilingData) const
    {
        uint32_t blockDim = 0;
        uint32_t maxUb = 0;
        uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
        // leave 20% ub for system
        uint64_t maxUbSize = static_cast<uint64_t>(PlatformInfo::Instance().GetUbSize() * 0.8);
        maxUbSize = (maxUbSize + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES; // align to 32
        for (size_t i = 0; i < MEMSET_MAX_TENSOR_NUM && i < memsetInfo.size(); ++i) {
            uint64_t size = (memsetInfo[i].size + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
            tilingData.size[i] = size;
            if (size < BLOCK_BYTES_ONCE * coreNum) {
                tilingData.sizeBlock[i] = size;
                tilingData.sizeLoop[i] = size;
                blockDim = std::max(blockDim, 1U);
                maxUb = std::max(maxUb, tilingData.sizeLoop[i]);
                MKI_LOG(INFO) << "Memset tiling " << i << " single core, size " << size;
                continue;
            }
            MKI_CHECK(coreNum > 0, "coreNum is 0", return 0);
            uint64_t sizeBlock = (size + coreNum - 1) / coreNum;
            uint64_t sizeBlockAligned = (sizeBlock + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
            uint64_t usedCore = (size + sizeBlockAligned - 1) / sizeBlockAligned;
            blockDim = std::max(blockDim, static_cast<uint32_t>(usedCore));
            tilingData.sizeBlock[i] = sizeBlockAligned;
            uint64_t sizeLoop = std::min(maxUbSize, sizeBlockAligned);
            tilingData.sizeLoop[i] = sizeLoop;
            maxUb = std::max(maxUb, static_cast<uint32_t>(sizeLoop));
            MKI_LOG(INFO) << "Memset tiling " << i << " blockDim " << usedCore << ", size " << size << ", sizeBlock "
                          << sizeBlockAligned << ", sizeLoop " << sizeLoop;
        }
        tilingData.maxUb[0] = maxUb;
        MKI_LOG(INFO) << "Memset tiling finished, blockDim " << blockDim << ", maxUb " << maxUb;
        return blockDim;
    }

    Status Run(void **args, uint64_t argsNum,
               MiniVector<KernelInfo::MemsetInfo> &memsetInfo, KernelHandle handle, void *stream)
    {
        MemsetArgs memsetArgs;
        (void)memset_s(&memsetArgs, sizeof(MemsetArgs), 0, sizeof(MemsetArgs));
        for (size_t i = 0; i < MEMSET_MAX_TENSOR_NUM && i < memsetInfo.size() && i < argsNum; ++i) {
            memsetArgs.tensors[i] = args[memsetInfo[i].argIdx];
        }

        uint32_t blockDim = MemsetTiling(memsetInfo, memsetArgs.tilingData);
        MKI_CHECK(blockDim > 0, "failed to run memset tiling", return Status::FailStatus(1));

        RtArgsExT argsEx;
        (void)memset_s(&argsEx, sizeof(RtArgsExT), 0, sizeof(RtArgsExT));
        argsEx.args = &memsetArgs;
        argsEx.argsSize = sizeof(MemsetArgs);
        argsEx.hasTiling = 1;
        argsEx.tilingAddrOffset = MEMSET_MAX_TENSOR_NUM * sizeof(void *);
        argsEx.tilingDataOffset = (MEMSET_MAX_TENSOR_NUM + 1) * sizeof(void *);

        MkiRtKernelParam kernelParam;
        (void)memset_s(&kernelParam, sizeof(MkiRtKernelParam), 0, sizeof(MkiRtKernelParam));
        kernelParam.blockDim = blockDim;
        kernelParam.argsEx = &argsEx;

        int st = MkiRtFunctionLaunchWithFlag(&handle, &kernelParam, stream, nullptr);
        MKI_CHECK(st == MKIRT_SUCCESS, "fail to launch memset", return Mki::Status::FailStatus(1));

        return Status::OkStatus();
    }
};

Status ClearTensors(void **args, uint64_t argsNum, MiniVector<KernelInfo::MemsetInfo> &memsetInfo, void *stream)
{
    static HandleRegister handleRegisterMemset("MemsetKernel");
    static MemsetKernel memsetKernel("MemsetKernel", handleRegisterMemset.GetHandle());
    return memsetKernel.Run(args, argsNum, memsetInfo, handleRegisterMemset.GetHandle(), stream);
}
} // namespace Mki
