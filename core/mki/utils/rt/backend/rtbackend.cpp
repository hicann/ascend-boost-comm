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
#include "mki/utils/rt/backend/rtbackend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/rt/backend/help_macro.h"

namespace Mki {
RtBackend::RtBackend() { Init(); }

RtBackend::~RtBackend()
{
    if (soHandle_) {
        dlclose(soHandle_);
        soHandle_ = nullptr;
    }
}

void RtBackend::Init()
{
    const char *asdHomePath = std::getenv("ASCEND_HOME_PATH");
    if (asdHomePath == nullptr) {
        std::cout << "env ASCEND_HOME_PATH not exist" << std::endl;
        initStatus_ = MKIRT_ERROR_ASCEND_ENV_NOT_EXIST;
        return;
    }

    std::string runtimeSoPath = std::string(asdHomePath) + "/lib64/libruntime.so";
    std::string realPath = FileSystem::PathCheckAndRegular(runtimeSoPath, false);
    MKI_CHECK(!realPath.empty(), "runtimeSoPath is null", return);
    soHandle_ = dlopen(realPath.c_str(), RTLD_LAZY);
    if (soHandle_ == nullptr) {
        std::cout << "dlopen fail, error:" << dlerror() << ", runtime so path:" << realPath << std::endl;
        initStatus_ = MKIRT_ERROR_LOAD_RUNTIME_FAIL;
        return;
    }

    InitDeviceFuncs();
    InitMemFuncs();
    InitModuleFuncs();
    InitStreamFuncs();
    InitOtherFuncs();
}

void RtBackend::InitDeviceFuncs()
{
    rtGetDeviceCount_ = (RtGetDeviceCountFunc)(dlsym(soHandle_, "rtGetDeviceCount"));
    rtGetDeviceIDs_ = (RtGetDeviceIDsFunc)dlsym(soHandle_, "rtGetDeviceIDs");
    rtSetDevice_ = (RtSetDeviceFunc)dlsym(soHandle_, "rtSetDevice");
    rtResetDevice_ = (RtResetDeviceFunc)dlsym(soHandle_, "rtDeviceReset");
    rtGetDevice_ = (RtGetDeviceFunc)dlsym(soHandle_, "rtGetDevice");
    rtSetSocVersion_ = (RtSetSocVersionFunc)dlsym(soHandle_, "rtSetSocVersion");
    rtGetSocVersion_ = (RtGetSocVersionFunc)dlsym(soHandle_, "rtGetSocVersion");
    rtDeviceGetBareTgid_ = (RtDeviceGetBareTgidFunc)dlsym(soHandle_, "rtDeviceGetBareTgid");
    rtDevicePairDevicesInfo_ = (RtDeviceGetPairDevicesInfoFunc)dlsym(soHandle_, "rtGetPairDevicesInfo");
    MKI_LOG(DEBUG) << "Rt GetDeviceCount Func:" << rtGetDeviceCount_
                  << ", Rt GetDeviceIDs Func:" << rtGetDeviceIDs_
                  << ", Rt SetDevice Func:" << rtSetDevice_
                  << ", Rt ResetDevice Func:" << rtResetDevice_
                  << ", Rt GetDevice Func:" << rtGetDevice_
                  << ", Rt SetSocVersion Func:" << rtSetSocVersion_
                  << ", Rt GetSocVersion Func:" << rtGetSocVersion_
                  << ", Rt DeviceGetBareTgid Func:" << rtDeviceGetBareTgid_
                  << ", Rt DeviceGetPairDevicesInfo Func:" << rtDevicePairDevicesInfo_;
}

void RtBackend::InitMemFuncs()
{
    rtMalloc_ = (RtMallocFunc)dlsym(soHandle_, "rtMalloc");
    rtFree_ = (RtFreeFunc)dlsym(soHandle_, "rtFree");
    rtMallocHost_ = (RtMallocHostFunc)dlsym(soHandle_, "rtMallocHost");
    rtFreeHost_ = (RtFreeHostFunc)dlsym(soHandle_, "rtFreeHost");
    rtMemcpy_ = (RtMemcpyFunc)dlsym(soHandle_, "rtMemcpy");
    rtMemcpyAsync_ = (RtMemcpyAsyncFunc)dlsym(soHandle_, "rtMemcpyAsync");
    rtMemsetAsync_ = (RtMemsetAsyncFunc)dlsym(soHandle_, "rtMemsetAsync");
    rtIpcSetMemoryName_ = (RtIpcSetMemoryNameFunc)dlsym(soHandle_, "rtIpcSetMemoryName");
    rtIpcOpenMemory_ = (RtIpcOpenMemoryFunc)dlsym(soHandle_, "rtIpcOpenMemory");
    rtSetIpcMemPid_ = (RtSetIpcMemPidFunc)dlsym(soHandle_, "rtSetIpcMemPid");
    MKI_LOG(DEBUG) << "Rt Malloc Func:" << rtMalloc_ << ", RtFreeFunc:" << rtFree_
                  << ", Rt MallocHost Func:" << rtMallocHost_ << ", RtFreeHostFunc:" << rtFreeHost_
                  << ", Rt Memcpy Func:" << rtMemcpy_ << ", RtMemcpyAsyncFunc:" << rtMemcpyAsync_
                  << ", Rt MemsetAsync Func: " << rtMemsetAsync_
                  << ", Rt IpcSetMemoryName Func: " << rtIpcSetMemoryName_
                  << ", Rt IpcOpenMemory Func: " << rtIpcOpenMemory_
                  << ", Rt SetIpcMemPid Func: " << rtSetIpcMemPid_;
}

void RtBackend::InitModuleFuncs()
{
    rtDevBinaryRegister_ = (RtDevBinaryRegisterFunc)dlsym(soHandle_, "rtDevBinaryRegister");
    rtDevBinaryUnRegister_ = (RtDevBinaryUnRegisterFunc)dlsym(soHandle_, "rtDevBinaryUnRegister");
    rtFunctionRegister_ = (RtFunctionRegisterFunc)dlsym(soHandle_, "rtFunctionRegister");
    rtRegisterAllKernel_ = (RtRegisterAllKernelFunc)dlsym(soHandle_, "rtRegisterAllKernel");
    rtKernelLaunch_ = (RtKernelLaunchFunc)dlsym(soHandle_, "rtKernelLaunch");
    rtKernelLaunchWithHandle_ = (RtKernelLaunchWithHandleFunc)dlsym(soHandle_, "rtKernelLaunchWithHandleV2");
    rtKernelLaunchWithFlag_ = (RtKernelLaunchWithFlagFunc)dlsym(soHandle_, "rtKernelLaunchWithFlagV2");
    MKI_LOG(DEBUG) << "Rt DevBinaryRegister Func:" << rtDevBinaryRegister_
                  << ", Rt DevBinaryUnRegister Func:" << rtDevBinaryUnRegister_
                  << ", Rt FunctionRegister Func:" << rtFunctionRegister_
                  << ", Rt RegisterAllKernel Func:" << rtRegisterAllKernel_
                  << ", Rt KernelLaunch Func:" << rtKernelLaunch_
                  << ", Rt KernelLaunchWithHandle Func:" << rtKernelLaunchWithHandle_
                  << ", Rt KernelLaunchWithFlag Func:"<< rtKernelLaunchWithFlag_;
}

void RtBackend::InitStreamFuncs()
{
    rtStreamCreate_ = (RtStreamCreateFunc)dlsym(soHandle_, "rtStreamCreate");
    rtStreamDestroy_ = (RtStreamDestroyFunc)dlsym(soHandle_, "rtStreamDestroy");
    rtStreamSynchronize_ = (RtStreamSynchronizeFunc)dlsym(soHandle_, "rtStreamSynchronize");
    rtGetStreamId_ = (RtGetStreamIdFunc)dlsym(soHandle_, "rtGetStreamId");
    MKI_LOG(DEBUG) << "Rt StreamCreate Func:" << rtStreamCreate_ <<
                     ", Rt StreamDestroy Func:" << rtStreamDestroy_ <<
                     ", Rt StreamSynchronize Func:" << rtStreamSynchronize_ <<
                     ", Rt GetStreamId Func:" << rtGetStreamId_;
}

void RtBackend::InitOtherFuncs()
{
    rtGetC2cCtrlAddr_ = (RtGetC2cCtrlAddrFunc)dlsym(soHandle_, "rtGetC2cCtrlAddr");
    MKI_LOG(DEBUG) << "Rt GetC2cCtrlAddr Func:" << rtGetC2cCtrlAddr_;
}

int RtBackend::DeviceGetCount(int32_t *devCount)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtGetDeviceCount_);
    MKI_LOG(INFO) << "Rt GetDeviceCount Func start";
    CHECK_STATUS_RETURN(rtGetDeviceCount_(devCount));
}

int RtBackend::DeviceGetIds(int32_t *devIds, int32_t devIdNum)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtGetDeviceIDs_);
    CHECK_STATUS_RETURN(rtGetDeviceIDs_(devIds, devIdNum));
}

int RtBackend::DeviceGetCurrent(int32_t *devId)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtGetDevice_);
    CHECK_STATUS_RETURN(rtGetDevice_(devId));
}

int RtBackend::DeviceSetCurrent(int32_t devId)
{
    CHECK_INITED_RETURN(initStatus_);
    MKI_LOG(INFO) << "rt SetDevice start, devId:" << devId;
    CHECK_FUNC_EIXST_RETURN(rtSetDevice_);
    CHECK_STATUS_WITH_DESC_RETURN(rtSetDevice_(devId), "rt SetDevice");
}

int RtBackend::DeviceResetCurrent(int32_t devId)
{
    CHECK_INITED_RETURN(initStatus_);
    MKI_LOG(INFO) << "rt ResetDevice start, devId:" << devId;
    CHECK_FUNC_EIXST_RETURN(rtResetDevice_);
    CHECK_STATUS_WITH_DESC_RETURN(rtResetDevice_(devId), "rt DeviceReset");
}

int RtBackend::DeviceSetSocVersion(const char *version)
{
    CHECK_INITED_RETURN(initStatus_);
    MKI_LOG(INFO) << "rt SetSocVersion start, version:" << version;
    CHECK_FUNC_EIXST_RETURN(rtSetSocVersion_);
    CHECK_STATUS_WITH_DESC_RETURN(rtSetSocVersion_(version), "rt SetSocVersion");
}

int RtBackend::DeviceGetSocVersion(char *version, uint32_t maxLen)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtGetSocVersion_);
    CHECK_STATUS_RETURN(rtGetSocVersion_(version, maxLen));
}

int RtBackend::DeviceGetBareTgid(uint32_t *pid)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtDeviceGetBareTgid_);
    CHECK_STATUS_RETURN(rtDeviceGetBareTgid_(pid));
}

int RtBackend::DeviceGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtDevicePairDevicesInfo_);
    CHECK_STATUS_RETURN(rtDevicePairDevicesInfo_(devId, otherDevId, infoType, val));
}

int RtBackend::StreamCreate(MkiRtStream *stream, int32_t priority)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtStreamCreate_);
    CHECK_STATUS_WITH_DESC_RETURN(rtStreamCreate_(stream, priority), "rt Create Stream");
}

int RtBackend::StreamDestroy(MkiRtStream stream)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtStreamDestroy_);
    CHECK_STATUS_RETURN(rtStreamDestroy_(stream));
}

int RtBackend::StreamSynchronize(MkiRtStream stream)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtStreamSynchronize_);
    CHECK_STATUS_RETURN(rtStreamSynchronize_(stream));
}

int RtBackend::StreamGetId(MkiRtStream stream, int32_t *streamId)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtGetStreamId_);
    CHECK_STATUS_WITH_DESC_RETURN(rtGetStreamId_(stream, streamId), "rt GetStreamId");
}

int RtBackend::MemMallocDevice(void **devPtr, uint64_t size, MkiRtMemType memType)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtMalloc_);
    MKI_LOG(INFO) << "rtMalloc start, size:" << size << ", memType:" << memType;
    CHECK_STATUS_WITH_DESC_RETURN(rtMalloc_(devPtr, size, memType), "rtMalloc");
}

int RtBackend::MemFreeDevice(void *devPtr)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtFree_);
    CHECK_STATUS_RETURN(rtFree_(devPtr));
}

int RtBackend::MemMallocHost(void **hostPtr, uint64_t size)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtMallocHost_);
    CHECK_STATUS_RETURN(rtMallocHost_(hostPtr, size));
}

int RtBackend::MemFreeHost(void *hostPtr)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtFreeHost_);
    CHECK_STATUS_RETURN(rtFreeHost_(hostPtr));
}

int RtBackend::MemCopy(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen, MkiRtMemCopyType copyType)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtMemcpy_);
    CHECK_STATUS_RETURN(rtMemcpy_(dst, dstLen, srcPtr, srcLen, copyType));
}

int RtBackend::MemCopyAsync(void *dst, uint64_t dstLen, const void *srcPtr, uint64_t srcLen,
                            MkiRtMemCopyType copyType, void *stream)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtMemcpyAsync_);
    CHECK_STATUS_RETURN(rtMemcpyAsync_(dst, dstLen, srcPtr, srcLen, copyType, stream));
}

int RtBackend::MemSetAsync(void *dst, uint64_t destMax, uint32_t value, uint64_t count, void *stream)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtMemsetAsync_);
    CHECK_STATUS_RETURN(rtMemsetAsync_(dst, destMax, value, count, stream));
}

int RtBackend::IpcSetMemoryName(const void *ptr, uint64_t byteCount, const char *name, uint32_t len)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtIpcSetMemoryName_);
    CHECK_STATUS_RETURN(rtIpcSetMemoryName_(ptr, byteCount, name, len));
}

int RtBackend::IpcOpenMemory(void **ptr, const char *name)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtIpcOpenMemory_);
    CHECK_STATUS_RETURN(rtIpcOpenMemory_(ptr, name));
}

int RtBackend::SetIpcMemPid(const char *name, int32_t pid[], int num)
{
    CHECK_INITED_RETURN(initStatus_);
    CHECK_FUNC_EIXST_RETURN(rtSetIpcMemPid_);
    CHECK_STATUS_RETURN(rtSetIpcMemPid_(name, pid, num));
}
}
