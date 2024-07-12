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
