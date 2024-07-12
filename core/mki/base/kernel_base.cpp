/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "mki/base/kernel_base.h"
#include <securec.h>
#include "mki/loader/device_kernel/device_kernel_loader.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/singleton/singleton.h"
#include "mki/utils/memset/memset_launcher.h"

#define UNUSED_VALUE(x) (void)(x)

namespace Mki {
KernelBase::KernelBase(const std::string &opName, KernelHandle handle) : kernelName_(opName), handle_(handle)
{
    launchBufferSize_ = GetSingleton<KernelBinaryLoader>().GetKernelTilingSize(kernelName_);
    MKI_LOG(DEBUG) << "Create kernel " << kernelName_ << ", launch buffer size " << launchBufferSize_;
    UpdateKernelType();
}

KernelBase::~KernelBase() {}

std::string KernelBase::GetName() const { return kernelName_; }

KernelType KernelBase::GetType() const { return kernelType_; }

KernelHandle KernelBase::GetHandle() const { return handle_; }

KernelInfo &KernelBase::GetKernelInfo() const { return kInfo_; }

void KernelBase::SetLaunchWithTiling(bool flag) { launchWithTiling_ = flag; }

bool KernelBase::GetLaunchWithTiling() const { return launchWithTiling_; }

void KernelBase::Reset() { initFlag_ = false; }

Status KernelBase::Init(const LaunchParam &launchParam)
{
    MKI_CHECK(CheckInTensors(launchParam), "Not supported in tensors", return Status::FailStatus(1));
    MKI_CHECK(CanSupport(launchParam), "Not supported op", return Status::FailStatus(1));

    GetKernelInfo().Reset(launchWithTiling_);

    auto tilingSize = GetTilingSize(launchParam);
    if (launchWithTiling_) {
        GetKernelInfo().AllocTilingHost(tilingSize);
    }

    auto status = InitImpl(launchParam);
    MKI_CHECK(status.Ok(), "Failed to init run info " << status.ToString(), return status);

    auto kernelParamNum = GetKernelParamNum(launchParam);
    uint64_t baseSize = kernelParamNum * sizeof(void *);
    uint64_t argsSize = baseSize;
    if (!launchWithTiling_) {
        return GetKernelInfo().InitArgs(argsSize);
    }

    uint64_t tilingUsedSize = Utils::GetTensorAlignedSize(GetKernelInfo().GetTilingUsedSize());
    argsSize += tilingUsedSize;
    uint64_t constTensorSize = GetKernelInfo().GetTilingSize() - GetKernelInfo().GetConstTensorOffset();
    argsSize += constTensorSize;
    MKI_LOG(INFO) << "args num " << kernelParamNum << ", tiling used size " << tilingUsedSize
                  << ", const tensor size " << constTensorSize;
    status = GetKernelInfo().InitArgs(argsSize);
    MKI_CHECK(status.Ok(), "failed to init args", return status);
    uint8_t *args = GetKernelInfo().GetArgs();
    auto ret = memcpy_s(args + baseSize, argsSize - baseSize,
                        GetKernelInfo().GetTilingHostAddr(), GetKernelInfo().GetTilingUsedSize());
    MKI_CHECK(ret == EOK, "failed to copy tiling", return Status::FailStatus(-1));
    MKI_LOG(INFO) << "copy tiling data " << tilingUsedSize << " to args offset " << baseSize;
    if (constTensorSize > 0) {
        ret = memcpy_s(args + baseSize + tilingUsedSize, argsSize - baseSize - tilingUsedSize,
                       GetKernelInfo().GetTilingHostAddr() + GetKernelInfo().GetConstTensorOffset(), constTensorSize);
        MKI_CHECK(ret == EOK, "failed to copy const tensor", return Status::FailStatus(-1));
        MKI_LOG(INFO) << "copy const data " << constTensorSize << " to args offset " << baseSize + tilingUsedSize;
    }
    return Status::OkStatus();
}

Status KernelBase::Run(const LaunchParam &launchParam, RunInfo &runInfo)
{
    RtArgsExT argsEx;
    uint64_t argsNum = GetKernelParamNum(launchParam);
    auto &kernelInfo = GetKernelInfo();
    uint8_t *argsPtr = kernelInfo.GetArgs();
    uint64_t argsSize = kernelInfo.GetArgsSize();
    MKI_CHECK(argsPtr != nullptr, "args size invalid", return Status::FailStatus(-1));
    MKI_CHECK(argsNum * sizeof(void *) <= argsSize, "args size invalid", return Status::FailStatus(-1));
    auto ret = memset_s(argsPtr, argsSize, 0, argsNum * sizeof(void *));
    MKI_CHECK(ret == EOK, "memory set failed", return Status::FailStatus(ERROR_INVALID_VALUE));
    void **args = reinterpret_cast<void **>(static_cast<void *>(argsPtr));
    // set hwsync
    auto status = UpdateHwsyncArgs(args, argsNum);
    MKI_CHECK(status.Ok(), "failed to update hwsync args", return status);
    // set const input
    size_t constTensorCount = kernelInfo.GetConstTensorCount();
    RtHostInputInfoT hostInfo[constTensorCount];
    status = launchWithTiling_ ? UpdateConstTensorArgs(args, argsNum, hostInfo, constTensorCount)
                                        : UpdateConstTensorArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to update const tensor args", return status);
    // set input / output / workspace
    status = UpdateInOutWkspArgs(args, argsNum, launchParam, runInfo);
    MKI_CHECK(status.Ok(), "failed to update input output wksp args", return status);
    // set tiling
    status = launchWithTiling_ ? UpdateTilingArgs(argsEx, argsNum, runInfo)
                                        : UpdateTilingArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to get launch with tiling", return status);
    // Memset
    status = MemsetTensorArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to memset tensor args", return status);
    // launch
    MkiRtKernelParam kernelParam{};
    kernelParam.tilingId = kernelInfo.GetTilingId();
    kernelParam.blockDim = kernelInfo.GetBlockDim();
    kernelParam.argsEx = &argsEx;
    argsEx.args = argsPtr;
    argsEx.argsSize = argsSize;
    if (launchWithTiling_) {
        argsEx.hostInputInfoPtr = hostInfo;
        argsEx.hostInputInfoNum = constTensorCount;
    }
    MKI_LOG(INFO) << "Ready to run, KernelInfo:\n" << kernelInfo.ToString();
    if (*handle_ != nullptr) {
        MKI_LOG(DEBUG) << "launch function with handle";
        int st = MkiRtFunctionLaunchWithHandle(*handle_, &kernelParam, runInfo.GetStream(), nullptr);
        MKI_CHECK(
            st == MKIRT_SUCCESS, "Mki RtFunction LaunchWithHandle fail",
            return Mki::Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR, "Mki RtFunction LaunchWithHandle fail"));
    } else {
        MKI_LOG(DEBUG) << "launch function with flag";
        int st = MkiRtFunctionLaunchWithFlag(handle_, &kernelParam, runInfo.GetStream(), nullptr);
        MKI_CHECK(st == MKIRT_SUCCESS, "Mki RtFunction LaunchWithFlag fail",
                    return Mki::Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR, "Mki RtFunction Launch fail"));
    }
    return Mki::Status::OkStatus();
}

bool KernelBase::CanSupport(const LaunchParam &launchParam) const
{
    UNUSED_VALUE(launchParam);
    MKI_CHECK(false, "CanSupport has to be overriden!", return false);
}

void KernelBase::Copy(const KernelBase &other)
{
    kernelName_ = other.kernelName_;
    launchBufferSize_ = other.launchBufferSize_;
    handle_ = other.handle_;
    kernelType_ = other.kernelType_;
    launchWithTiling_ = other.launchWithTiling_;
    initFlag_ = other.initFlag_;
    creator_ = other.creator_;
    kInfo_.Copy(other.kInfo_);
}

uint64_t KernelBase::GetTilingSize(const LaunchParam &launchParam) const
{
    (void)launchParam;
    return launchBufferSize_;
}

Status KernelBase::InitImpl(const LaunchParam &launchParam)
{
    UNUSED_VALUE(launchParam);
    MKI_CHECK(false, "InitImpl has to be overriden!", return Status::FailStatus(-1));
}

uint64_t KernelBase::GetKernelParamNum(const LaunchParam &launchParam)
{
    uint64_t inputOutputNum = launchParam.GetInTensorCount() + launchParam.GetOutTensorCount();
    uint64_t constInputNum = kInfo_.GetConstTensorCount();
    uint64_t workspaceNum = kInfo_.GetScratchSizes().size();
    uint64_t hwsyncNum = kInfo_.GetHwsyncIdx() < 0 ? 0 : 1;
    MKI_LOG(DEBUG) << "kernel param: " << inputOutputNum << " in/out, " << constInputNum << " const in, "
                  << workspaceNum << " workspaces, " << hwsyncNum << " hwsync";
    return inputOutputNum + constInputNum + workspaceNum + hwsyncNum + 1; // 1 is tiling
}

void KernelBase::UpdateKernelType()
{
    int32_t coreType = GetSingleton<KernelBinaryLoader>().GetKernelCoreType(kernelName_);
    switch (coreType) {
        case -1: MKI_LOG(ERROR) << "Failed to get core type!"; return;  // -1 ERROR
        case 0: kernelType_ = KernelType::KERNEL_TYPE_AI_CORE; return;  // 0 AI core
        case 2: kernelType_ = KernelType::KERNEL_TYPE_AIV; return;      // 2 AIV core
        case 4: kernelType_ = KernelType::KERNEL_TYPE_MIX_AIC; return;  // 4 MIX AIC core
        default:
            MKI_LOG(WARN) << "Unexpected core type, use AIC as default!";
            kernelType_ = KernelType::KERNEL_TYPE_MIX_AIC;
            return;
    }
}

Status KernelBase::UpdateHwsyncArgs(void **args, uint64_t argsNum)
{
    int64_t hwsyncIdx = kInfo_.GetHwsyncIdx();
    if (hwsyncIdx >= 0 && static_cast<uint64_t>(hwsyncIdx) < argsNum) {
        uint64_t *addr = nullptr;
        uint32_t len = 0;
        int st = MkiRtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&addr), &len);
        MKI_CHECK(st == MKIRT_SUCCESS && addr != nullptr,
            "Mki Get RtC2cCtrlAddr fail", return Status::FailStatus(-1));
        MKI_LOG(INFO) << "args info: hwsync " << hwsyncIdx;
        *(args + static_cast<uint64_t>(hwsyncIdx)) = addr;
    }
    return Status::OkStatus();
}

Status KernelBase::UpdateConstTensorArgs(void **args, uint64_t argsNum,
                                         RtHostInputInfoT *info, uint64_t infoNum)
{
    auto &kernelInfo = kInfo_;
    uint64_t offset = Utils::GetTensorAlignedSize(kernelInfo.GetTilingUsedSize()) + argsNum * sizeof(void *);
    for (uint64_t i = 0; i < infoNum; i++) {
        auto &constTensorInfo = kernelInfo.GetConstTensorInfo(i);
        args[constTensorInfo.argIdx] = info + i; // placeholder
        info[i].addrOffset = constTensorInfo.argIdx * sizeof(void *);
        info[i].dataOffset = offset;
        MKI_LOG(DEBUG) << "args info: const tensor " << constTensorInfo.argIdx << " offset in args " << offset;
        offset += constTensorInfo.size;
    }
    return Status::OkStatus();
}

Status KernelBase::UpdateConstTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo)
{
    UNUSED_VALUE(argsNum);
    auto &kernelInfo = kInfo_;
    uint8_t *tilingDeviceAddr = runInfo.GetTilingDeviceAddr();
    uint64_t offset = kernelInfo.GetConstTensorOffset();
    for (size_t i = 0; i < kernelInfo.GetConstTensorCount(); i++) {
        auto &constTensorInfo = kernelInfo.GetConstTensorInfo(i);
        args[constTensorInfo.argIdx] = tilingDeviceAddr + offset;
        MKI_LOG(DEBUG) << "args info: const tensor " << constTensorInfo.argIdx << " offset in tiling " << offset;
        offset += constTensorInfo.size;
    }
    return Status::OkStatus();
}

Status KernelBase::UpdateInOutWkspArgs(void **args, uint64_t argsNum,
    const LaunchParam &launchParam, const RunInfo &runInfo)
{
    size_t inputNum = launchParam.GetInTensorCount();
    uint64_t idx = 0;
    for (size_t i = 0; i < inputNum && idx < argsNum; idx++) {
        if (args[idx] != nullptr) {
            continue;
        }
        MKI_LOG(DEBUG) << "args info: input tensor " << idx;
        args[idx] = launchParam.GetInTensor(i++).data;
    }
    size_t outputNum = launchParam.GetOutTensorCount();
    for (size_t i = 0; i < outputNum && idx < argsNum; idx++) {
        if (args[idx] != nullptr) {
            continue;
        }
        MKI_LOG(DEBUG) << "args info: output tensor " << idx;
        args[idx] = launchParam.GetOutTensor(i++).data;
    }
    auto workspaceAddr = runInfo.GetScratchDeviceAddr();
    auto &workspaces = kInfo_.GetScratchSizes();
    size_t workspaceNum = workspaces.size();
    uint64_t offset = 0;
    for (size_t i = 0; i < workspaceNum && idx < argsNum; idx++) {
        if (args[idx] != nullptr) {
            continue;
        }
        MKI_LOG(DEBUG) << "args info: workspace " << idx << " offset " << offset;
        args[idx] = workspaceAddr + offset;
        offset += workspaces.at(i++);
    }
    return Status::OkStatus();
}

Status KernelBase::UpdateTilingArgs(RtArgsExT &argsEx, uint64_t argsNum, RunInfo &runInfo) const
{
    UNUSED_VALUE(runInfo);
    MKI_CHECK((argsNum > 1), "argsNum invalid : " << argsNum, return Status::FailStatus(-1));
    MKI_LOG(DEBUG) << "argsNum: tiling " << (argsNum - 1);
    argsEx.hasTiling = 1;
    argsEx.tilingAddrOffset = (argsNum - 1) * sizeof(void *);
    argsEx.tilingDataOffset = argsNum * sizeof(void *);
    return Status::OkStatus();
}

Status KernelBase::UpdateTilingArgs(void **args, uint64_t argsNum, RunInfo &runInfo) const
{
    MKI_LOG(DEBUG) << "args info: tiling " << (argsNum - 1);
    args[argsNum - 1] = runInfo.GetTilingDeviceAddr();
    return Status::OkStatus();
}

Status KernelBase::MemsetTensorArgs(void **args, uint64_t argsNum, const RunInfo &runInfo)
{
    auto &kernelInfo = kInfo_;
    if (kernelInfo.GetMemsetInfo().size() != 0) {
        Status status = ClearTensors(args, argsNum, kernelInfo.GetMemsetInfo(), runInfo.GetStream());
        MKI_CHECK(status.Ok(), "failed to clear tensors", return status);
    }
    return Status::OkStatus();
}

Kernel *KernelBase::Clone() const
{
    MKI_CHECK(creator_ != nullptr, kernelName_ << " creator is nullptr", return nullptr);
    KernelBase *kernel = creator_();
    kernel->Copy(*this);
    return kernel;
}

} // namespace Mki
