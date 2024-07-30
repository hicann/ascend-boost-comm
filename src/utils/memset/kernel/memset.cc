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

#include "kernel_operator.h"

namespace {
constexpr uint32_t TILING_LEN = 4 * 8; // 4 * 8 uint32 list

class Memset {
public:
    __aicore__ inline Memset() {}

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
}

extern "C" __global__ __aicore__ void memory_set(GM_ADDR tensor0, GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3,
                                             GM_ADDR tensor4, GM_ADDR tensor5, GM_ADDR tensor6, GM_ADDR tensor7,
                                             GM_ADDR tiling)
{
    Memset kernel;
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
