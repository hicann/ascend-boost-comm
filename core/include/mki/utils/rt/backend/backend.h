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
#ifndef MKIRT_BACKEND_BACKEND_H
#define MKIRT_BACKEND_BACKEND_H
#include "mki/utils/rt/base/types.h"

namespace Mki {
class Backend {
public:
    Backend();
    virtual ~Backend();

public:
    virtual int DeviceGetCount(int32_t *devCount) = 0;
    virtual int DeviceGetIds(int32_t *devIds, int32_t devIdNum) = 0;
    virtual int DeviceGetCurrent(int32_t *devId) = 0;
    virtual int DeviceSetCurrent(int32_t devId) = 0;
    virtual int DeviceResetCurrent(int32_t devId) = 0;
    virtual int DeviceSetSocVersion(const char *version) = 0;
    virtual int DeviceGetSocVersion(char *version, uint32_t maxLen) = 0;
    virtual int DeviceGetBareTgid(uint32_t *pid) = 0;
    virtual int DeviceGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val) = 0;

public:
    virtual int StreamCreate(MkiRtStream *stream, int32_t priority) = 0;
    virtual int StreamDestroy(MkiRtStream stream) = 0;
    virtual int StreamSynchronize(MkiRtStream stream) = 0;
    virtual int StreamGetId(MkiRtStream stream, int32_t *streamId) = 0;

public:
    virtual int MemMallocDevice(void **devPtr, uint64_t size, MkiRtMemType memType = MKIRT_MEM_DEFAULT) = 0;
    virtual int MemFreeDevice(void *devPtr) = 0;
    virtual int MemMallocHost(void **hostPtr, uint64_t size) = 0;
    virtual int MemFreeHost(void *hostPtr) = 0;
    virtual int MemCopy(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen,
                        MkiRtMemCopyType copyType) = 0;
    virtual int MemCopyAsync(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen,
                             MkiRtMemCopyType copyType, void *stream) = 0;
    virtual int MemSetAsync(void *dst, uint64_t destMax, uint32_t value, uint64_t count, void *stream) = 0;
    virtual int IpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len) = 0;
    virtual int IpcOpenMemory(void **ptr, const char *name) = 0;
    virtual int SetIpcMemPid(const char *name, int32_t pid[], int num) = 0;

public:
    virtual int ModuleCreate(MkiRtModuleInfo *moduleInfo, MkiRtModule *module) = 0;
    virtual int ModuleCreateFromFile(const char *moduleFilePath, MkiRtModuleType type, int version,
                                     MkiRtModule *module) = 0;
    virtual int ModuleDestory(MkiRtModule *module) = 0;
    virtual int ModuleBindFunction(MkiRtModule module, const char *funcName, void *func) = 0;
    virtual int RegisterAllFunction(MkiRtModuleInfo *moduleInfo, void **handle) = 0;
    virtual int FunctionLaunch(const void *func, const MkiRtKernelParam *param, MkiRtStream stream) = 0;
    virtual int FunctionLaunchWithHandle(void *handle, const MkiRtKernelParam *param, MkiRtStream stream,
                                         const RtTaskCfgInfoT *cfgInfo) = 0;
    virtual int FunctionLaunchWithFlag(const void *func, const MkiRtKernelParam *param, MkiRtStream stream,
                                       const RtTaskCfgInfoT *cfgInfo) = 0;

public:
    virtual int GetC2cCtrlAddr(uint64_t *addr, uint32_t *len) = 0;
};
}

#endif
