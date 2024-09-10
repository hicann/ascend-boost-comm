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
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UNPAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UNPAD_H_

#include <mki/launch_param.h>
#include <mki/utils/status/status.h>
#include <mki/kernel_info.h>

namespace AtbOps {
using namespace Mki;
Status UnpadTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo);
} // namespace AtbOps
#endif
