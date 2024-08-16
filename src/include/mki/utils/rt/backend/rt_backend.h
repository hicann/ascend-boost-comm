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
#ifndef MKI_UTILS_RT_BACKEND_RTBACKEND_H
#define MKI_UTILS_RT_BACKEND_RTBACKEND_H
#include "mki/utils/rt/backend/backend.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t magic{0};
    uint32_t version{0};
    const void *data{nullptr};
    uint64_t length{0};
} RtDevBinaryT;

typedef void *rtStream_t;
int rtGetDeviceCount(int32_t *cnt);
int rtGetDeviceIDs(int32_t *devices, int32_t len);
int rtSetDevice(int32_t devId);
int rtDeviceReset(int32_t devId);
int rtGetDevice(int32_t *devId);
int rtSetSocVersion(const char *version);
int rtGetSocVersion(char *version, uint32_t maxLen);
int rtDeviceGetBareTgid(uint32_t *pid);
int rtGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val);
 
// rt stream
typedef void *rtStream_t;
int rtStreamCreate(rtStream_t *stm, int32_t priority);
int rtStreamDestroy(rtStream_t stm);
int rtStreamSynchronize(rtStream_t stm);
int rtGetStreamId(rtStream_t stm, int32_t *streamId);
 
// rt mem
int rtMalloc(void **devPtr, uint64_t size, uint32_t type);
int rtFree(void *devPtr);
int rtMallocHost(void **hostPtr, uint64_t size);
int rtFreeHost(void *hostPtr);
int rtMemcpy(void *dst, uint64_t destMax, const void *src, uint64_t cnt, int kind);
int rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t cnt, int kind, void *stream);
int rtMemsetAsync(void *dst, uint64_t destMax, uint32_t value, uint64_t cnt, void *stream);
int rtIpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len);
int rtIpcOpenMemory(void **ptr, const char *name);
int rtSetIpcMemPid(const char *name, int32_t pid[], int num);
 
int rtDevBinaryRegister(const RtDevBinaryT *bin, void **hdl);
int rtDevBinaryUnRegister(void *hdl);
int rtFunctionRegister(void *binHandle, const void *subFunc, const char *stubName,
                                   const void *kernelInfoExt, uint32_t funcMode);
int rtRegisterAllKernel(const RtDevBinaryT *bin, void **hdl);
int rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize, void *smDesc,
                               rtStream_t sm);
int rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
                                        RtArgsExT *argsInfo, void *smDesc, rtStream_t sm,
                                        const RtTaskCfgInfoT *cfgInfo);
int rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, RtArgsExT *argsInfo, void *smDesc,
                                        rtStream_t sm, uint32_t flags, const RtTaskCfgInfoT *cfgInfo);
 
// rt other
int rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

#ifdef __cplusplus
}
#endif

namespace Mki {
class RtBackend : public Backend {
public:
    RtBackend();
    ~RtBackend() override;

public:
    int DeviceGetCount(int32_t *devCount) override;
    int DeviceGetIds(int32_t *devIds, int32_t devIdNum) override;
    int DeviceGetCurrent(int32_t *devId) override;
    int DeviceSetCurrent(int32_t devId) override;
    int DeviceResetCurrent(int32_t devId) override;
    int DeviceSetSocVersion(const char *version) override;
    int DeviceGetSocVersion(char *version, uint32_t maxLen) override;
    int DeviceGetBareTgid(uint32_t *pid) override;
    int DeviceGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val) override;

public:
    int StreamCreate(MkiRtStream *stream, int32_t priority) override;
    int StreamDestroy(MkiRtStream stream) override;
    int StreamSynchronize(MkiRtStream stream) override;
    int StreamGetId(MkiRtStream stream, int32_t *streamId) override;

public:
    int MemMallocDevice(void **devPtr, uint64_t size, MkiRtMemType memType) override;
    int MemFreeDevice(void *devPtr) override;
    int MemMallocHost(void **hostPtr, uint64_t size) override;
    int MemFreeHost(void *hostPtr) override;
    int MemCopy(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen,
                        MkiRtMemCopyType copyType) override;
    int MemCopyAsync(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen,
                     MkiRtMemCopyType copyType, void *stream) override;
    int MemSetAsync(void *dst, uint64_t destMax, uint32_t value, uint64_t count, void *stream) override;
    int IpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len) override;
    int IpcOpenMemory(void **ptr, const char *name) override;
    int SetIpcMemPid(const char *name, int32_t pid[], int num) override;

public:
    int ModuleCreate(MkiRtModuleInfo *moduleInfo, MkiRtModule *module) override;
    int ModuleCreateFromFile(const char *moduleFilePath, MkiRtModuleType type, int version,
                             MkiRtModule *module) override;
    int ModuleDestory(MkiRtModule *module) override;
    int ModuleBindFunction(MkiRtModule module, const char *funcName, void *func) override;
    int RegisterAllFunction(MkiRtModuleInfo *moduleInfo, void **handle) override;
    int FunctionLaunch(const void *func, const MkiRtKernelParam *param, MkiRtStream stream) override;
    int FunctionLaunchWithHandle(void *handle, const MkiRtKernelParam *param, MkiRtStream stream,
                                 const RtTaskCfgInfoT *cfgInfo) override;
    int FunctionLaunchWithFlag(const void *func, const MkiRtKernelParam *param, MkiRtStream stream,
                               const RtTaskCfgInfoT *cfgInfo) override;

public:
    int GetC2cCtrlAddr(uint64_t *addr, uint32_t *len) override;

private:
    RtBackend(const RtBackend &) = delete;
    const RtBackend &operator=(const RtBackend &) = delete;
    int ModuleDestoryRtModule(void *rtModule);
};
}

#endif
