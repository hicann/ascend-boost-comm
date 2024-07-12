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
#include "mki/utils/rt/memory/memory.h"
#include "mki/utils/rt/backend/backend_factory.h"

namespace Mki {
int MkiRtMemMallocDevice(void **devPtr, uint64_t size, MkiRtMemType memType)
{
    return BackendFactory::GetBackend()->MemMallocDevice(devPtr, size, memType);
}

int MkiRtMemFreeDevice(void *devPtr) { return BackendFactory::GetBackend()->MemFreeDevice(devPtr); }

int MkiRtMemMallocHost(void **hostPtr, uint64_t size)
{
    return BackendFactory::GetBackend()->MemMallocHost(hostPtr, size);
}

int MkiRtMemFreeHost(void *hostPtr) { return BackendFactory::GetBackend()->MemFreeHost(hostPtr); }

int MkiRtMemCopy(void *dstPtr, uint64_t dstLen, const void *srcPtr, uint64_t srcLen, MkiRtMemCopyType copyType)
{
    return BackendFactory::GetBackend()->MemCopy(dstPtr, dstLen, srcPtr, srcLen, copyType);
}

int MkiRtMemCopyAsync(void *dstPtr, uint64_t dstLen, const void *srcPtr, uint64_t srcLen, MkiRtMemCopyType copyType,
                      void *stream)
{
    return BackendFactory::GetBackend()->MemCopyAsync(dstPtr, dstLen, srcPtr, srcLen, copyType, stream);
}

int MkiRtMemSetAsync(void *dstPtr, uint64_t destMax, uint32_t value, uint64_t count, void *stream)
{
    return BackendFactory::GetBackend()->MemSetAsync(dstPtr, destMax, value, count, stream);
}

int MkiRtIpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len)
{
    return BackendFactory::GetBackend()->IpcSetMemoryName(ptr, byteCount, name, len);
}

int MkiRtIpcOpenMemory(void **ptr, const char *name)
{
    return BackendFactory::GetBackend()->IpcOpenMemory(ptr, name);
}

int MkiRtSetIpcMemPid(const char *name, int32_t pid[], int num)
{
    return BackendFactory::GetBackend()->SetIpcMemPid(name, pid, num);
}
}
