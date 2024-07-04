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

#include "kernel_operator.h"

namespace Mki {

constexpr uint32_t TILING_LEN = 4 * 8; // 4 * 8 uint32 list

class MemsetKernel {
public:
    __aicore__ inline MemsetKernel() {}

    __aicore__ inline void Init(__gm__ uint8_t *tiling)
    {
        AscendC::GlobalTensor<uint32_t> tensorTilingGm;
        tensorTilingGm.SetGlobalBuffer((__gm__ uint32_t *)tiling);
        pipe_.InitBuffer(tilingBuf_, sizeof(uint32_t) * TILING_LEN);
        tilingTensor_ = tilingBuf_.Get<uint32_t>();
        DataCopy(tilingTensor_, tensorTilingGm, TILING_LEN);
        pipe_barrier(PIPE_ALL);

        maxUbSize_ = tilingTensor_.GetValue(3 * 8); // ub size offset, 3 list, each has 8 uint32
        pipe_barrier(PIPE_ALL);
        pipe_.InitBuffer(zeroBuf_, maxUbSize_);
        zeroTensor_ = zeroBuf_.Get<uint16_t>();
        Duplicate(zeroTensor_, (uint16_t)0, maxUbSize_ / sizeof(uint16_t));
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void CleanTensor(uint32_t idx, __gm__ uint8_t *tensor, __gm__ uint8_t *tiling)
    {
        if (tensor == nullptr) {
            return;
        }
        uint32_t size = tilingTensor_.GetValue(idx);
        uint32_t sizeBlock = tilingTensor_.GetValue(8 + idx); // 8: offset of block
        uint32_t sizeLoop = tilingTensor_.GetValue(8 * 2 + idx); // 8 * 2: offset of loop

        uint32_t base = sizeBlock * AscendC::GetBlockIdx();
        AscendC::GlobalTensor<uint16_t> tensorGm;
        for (uint32_t i = 0; i < sizeBlock; i += sizeLoop) {
            uint32_t leftCount = size - (base + i);
            uint32_t handleCount = leftCount < sizeLoop ? leftCount : sizeLoop;
            tensorGm.SetGlobalBuffer((__gm__ uint16_t *)(tensor + base + i));
            DataCopy(tensorGm, zeroTensor_, handleCount / sizeof(uint16_t));
        }
    }

private:
    uint32_t maxUbSize_{0};
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tilingBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> zeroBuf_;
    AscendC::LocalTensor<uint32_t> tilingTensor_;
    AscendC::LocalTensor<uint16_t> zeroTensor_;
};

extern "C" __global__ __aicore__ void memset(GM_ADDR tensor0, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3,
                                             GM_ADDR tensor4, GM_ADDR tensor5, GM_ADDR tensor6, GM_ADDR tensor7,
                                             GM_ADDR tiling)
{
    MemsetKernel kernel;
    kernel.Init(tiling);
    uint32_t idx = 0;
    kernel.CleanTensor(idx++, tensor0, tiling);
    kernel.CleanTensor(idx++, tensor1, tiling);
    kernel.CleanTensor(idx++, tensor2, tiling);
    kernel.CleanTensor(idx++, tensor3, tiling);
    kernel.CleanTensor(idx++, tensor4, tiling);
    kernel.CleanTensor(idx++, tensor5, tiling);
    kernel.CleanTensor(idx++, tensor6, tiling);
    kernel.CleanTensor(idx++, tensor7, tiling);
    pipe_barrier(PIPE_ALL);
}

} // Mki namespace
