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
#include "mki/utils/rt/stream/stream.h"
#include "mki/utils/rt/backend/backend_factory.h"

namespace Mki {
int MkiRtStreamCreate(MkiRtStream *stream, int32_t priority)
{
    return BackendFactory::GetBackend()->StreamCreate(stream, priority);
}

int MkiRtStreamDestroy(MkiRtStream stream) { return BackendFactory::GetBackend()->StreamDestroy(stream); }

int MkiRtStreamSynchronize(MkiRtStream stream)
{
    return BackendFactory::GetBackend()->StreamSynchronize(stream);
}

int MkiRtStreamGetId(MkiRtStream stream, int32_t *streamId)
{
    return BackendFactory::GetBackend()->StreamGetId(stream, streamId);
}
}
