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
