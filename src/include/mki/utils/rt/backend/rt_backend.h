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

// rt device
using RtGetDeviceCountFunc = int (*)(int32_t *cnt);
using RtGetDeviceIDsFunc = int (*)(int32_t *devices, int32_t len);
using RtSetDeviceFunc = int (*)(int32_t devId);
using RtResetDeviceFunc = int (*)(int32_t devId);
using RtGetDeviceFunc = int (*)(int32_t *devId);
using RtSetSocVersionFunc = int (*)(const char *version);
using RtGetSocVersionFunc = int (*)(char *version, uint32_t maxLen);
using RtDeviceGetBareTgidFunc = int(*)(uint32_t *pid);
using RtDeviceGetPairDevicesInfoFunc = int(*)(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val);

// rt stream
typedef void *rtStream_t;
using RtStreamCreateFunc = int (*)(rtStream_t *stm, int32_t priority);
using RtStreamDestroyFunc = int (*)(rtStream_t stm);
using RtStreamSynchronizeFunc = int (*)(rtStream_t stm);
using RtGetStreamIdFunc = int (*)(rtStream_t stm, int32_t *streamId);

// rt mem
using RtMallocFunc = int (*)(void **devPtr, uint64_t size, uint32_t type);
using RtFreeFunc = int (*)(void *devPtr);
using RtMallocHostFunc = int (*)(void **hostPtr, uint64_t size);
using RtFreeHostFunc = int (*)(void *hostPtr);
using RtMemcpyFunc = int (*)(void *dst, uint64_t destMax, const void *src, uint64_t cnt, int kind);
using RtMemcpyAsyncFunc = int(*)(void *dst, uint64_t destMax, const void *src, uint64_t cnt, int kind, void *stream);
using RtMemsetAsyncFunc = int(*)(void *dst, uint64_t destMax, uint32_t value, uint64_t cnt, void *stream);
using RtIpcSetMemoryNameFunc = int(*)(const void *ptr, uint64_t byteCount, const char *name, uint32_t len);
using RtIpcOpenMemoryFunc = int(*)(void **ptr, const char *name);
using RtSetIpcMemPidFunc = int(*)(const char *name, int32_t pid[], int num);

// rt kernel
typedef struct {
    uint32_t magic{0};
    uint32_t version{0};
    const void *data{nullptr};
    uint64_t length{0};
} RtDevBinaryT;

typedef void *rtStream_t;
using RtDevBinaryRegisterFunc = int (*)(const RtDevBinaryT *bin, void **hdl);
using RtDevBinaryUnRegisterFunc = int (*)(void *hdl);
using RtFunctionRegisterFunc = int (*)(void *binHandle, const void *subFunc, const char *stubName,
                                   const void *kernelInfoExt, uint32_t funcMode);
using RtRegisterAllKernelFunc = int (*)(const RtDevBinaryT *bin, void **hdl);
using RtKernelLaunchFunc = int (*)(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                   void *smDesc, rtStream_t sm);
using RtKernelLaunchWithHandleFunc = int(*)(void *hdl, const uint64_t tilingKey, uint32_t blockDim,
                                        RtArgsExT *argsInfo, void *smDesc, rtStream_t sm,
                                        const RtTaskCfgInfoT *cfgInfo);
using RtKernelLaunchWithFlagFunc = int(*)(const void *stubFunc, uint32_t blockDim, RtArgsExT *argsInfo, void *smDesc,
                                        rtStream_t sm, uint32_t flags, const RtTaskCfgInfoT *cfgInfo);

// rt other
using RtGetC2cCtrlAddrFunc = int (*)(uint64_t *addr, uint32_t *len);

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
    void Init();
    void InitDeviceFuncs();
    void InitMemFuncs();
    void InitModuleFuncs();
    void InitStreamFuncs();
    void InitOtherFuncs();
    int ModuleDestoryRtModule(void *rtModule);

private:
    void *soHandle_ = nullptr;
    int initStatus_ = MKIRT_SUCCESS;

private:
    RtGetDeviceCountFunc rtGetDeviceCount_ = nullptr;
    RtGetDeviceIDsFunc rtGetDeviceIDs_ = nullptr;
    RtSetDeviceFunc rtSetDevice_ = nullptr;
    RtResetDeviceFunc rtResetDevice_ = nullptr;
    RtGetDeviceFunc rtGetDevice_ = nullptr;
    RtSetSocVersionFunc rtSetSocVersion_ = nullptr;
    RtGetSocVersionFunc rtGetSocVersion_ = nullptr;
    RtDeviceGetBareTgidFunc rtDeviceGetBareTgid_ = nullptr;
    RtDeviceGetPairDevicesInfoFunc rtDevicePairDevicesInfo_ = nullptr;

private:
    RtStreamCreateFunc rtStreamCreate_ = nullptr;
    RtStreamDestroyFunc rtStreamDestroy_ = nullptr;
    RtStreamSynchronizeFunc rtStreamSynchronize_ = nullptr;
    RtGetStreamIdFunc rtGetStreamId_ = nullptr;

private:
    RtMallocFunc rtMalloc_ = nullptr;
    RtFreeFunc rtFree_ = nullptr;
    RtMallocHostFunc rtMallocHost_ = nullptr;
    RtFreeHostFunc rtFreeHost_ = nullptr;
    RtMemcpyFunc rtMemcpy_ = nullptr;
    RtMemcpyAsyncFunc rtMemcpyAsync_ = nullptr;
    RtMemsetAsyncFunc rtMemsetAsync_ = nullptr;
    RtIpcSetMemoryNameFunc rtIpcSetMemoryName_ = nullptr;
    RtIpcOpenMemoryFunc rtIpcOpenMemory_ = nullptr;
    RtSetIpcMemPidFunc rtSetIpcMemPid_ = nullptr;

private:
    RtDevBinaryRegisterFunc rtDevBinaryRegister_ = nullptr;
    RtDevBinaryUnRegisterFunc rtDevBinaryUnRegister_ = nullptr;
    RtFunctionRegisterFunc rtFunctionRegister_ = nullptr;
    RtRegisterAllKernelFunc rtRegisterAllKernel_ = nullptr;
    RtKernelLaunchFunc rtKernelLaunch_ = nullptr;
    RtKernelLaunchWithHandleFunc rtKernelLaunchWithHandle_ = nullptr;
    RtKernelLaunchWithFlagFunc rtKernelLaunchWithFlag_ = nullptr;

private:
    RtGetC2cCtrlAddrFunc rtGetC2cCtrlAddr_ = nullptr;
};
}

#endif
