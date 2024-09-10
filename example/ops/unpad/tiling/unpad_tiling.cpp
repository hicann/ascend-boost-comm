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
#include "unpad_tiling.h"
#include <mki/kernel_info.h>
#include "tiling_data.h"

namespace AtbOps {
using namespace Mki;
void FillTilingParam(const LaunchParam &launchParam, UnpadTilingData *tilingDataPtr)
{
    tilingDataPtr->padLength = launchParam.GetInTensor(0).desc.dims[1];
    tilingDataPtr->batch = launchParam.GetInTensor(0).desc.dims[0];
}

Status UnpadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    UnpadTilingData *tilingDataPtr = reinterpret_cast<UnpadTilingData *>(kernelInfo.GetTilingHostAddr());
    FillTilingParam(launchParam, tilingDataPtr);
    kernelInfo.SetBlockDim(1);

    uint64_t sysWorkspaceSize = 16;
    kernelInfo.GetScratchSizes() = {sysWorkspaceSize};
    return Status::OkStatus();
}
} // namespace AtbOps