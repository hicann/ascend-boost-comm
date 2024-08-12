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
#ifndef MKI_UTILS_RT_STREAM_STREAM_H
#define MKI_UTILS_RT_STREAM_STREAM_H
#include "mki/utils/rt/base/types.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace Mki {
int MkiRtStreamCreate(MkiRtStream *stream, int32_t priority = 0);
int MkiRtStreamDestroy(MkiRtStream stream);
int MkiRtStreamSynchronize(MkiRtStream stream);
int MkiRtStreamGetId(MkiRtStream stream, int32_t *streamId);
}
#ifdef __cplusplus
}
#endif
#endif
