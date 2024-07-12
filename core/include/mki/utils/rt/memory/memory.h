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
#ifndef MKI_UTILS_RT_MEMORY_MEMORY_H
#define MKI_UTILS_RT_MEMORY_MEMORY_H
#include "mki/utils/rt/base/types.h"
#ifdef __cplusplus
extern "C" {
#endif
namespace Mki {
int MkiRtMemMallocDevice(void **devPtr, uint64_t size, MkiRtMemType memType);
int MkiRtMemFreeDevice(void *devPtr);
int MkiRtMemMallocHost(void **hostPtr, uint64_t size);
int MkiRtMemFreeHost(void *hostPtr);
int MkiRtMemCopy(void *dstPtr, uint64_t dstLen, const void *srcPtr, uint64_t srcLen, MkiRtMemCopyType copyType);
int MkiRtMemCopyAsync(void *dstPtr, uint64_t dstLen, const void *srcPtr, uint64_t srcLen, MkiRtMemCopyType copyType,
                      void *stream);
int MkiRtMemSetAsync(void *dstPtr, uint64_t destMax, uint32_t value, uint64_t count, void *stream);
int MkiRtIpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len);
int MkiRtIpcOpenMemory(void **ptr, const char *name);
int MkiRtSetIpcMemPid(const char *name, int32_t pid[], int num);
}
#ifdef __cplusplus
}
#endif
#endif
