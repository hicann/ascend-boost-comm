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
#include "mki/base/kernel_base.h"
#include <securec.h>
#include "mki/utils/assert/assert.h"
#include "mki/utils/checktensor/check_tensor.h"
#include "mki/utils/file_system/file_system.h"
#include "mki/utils/log/log.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/memset/clear_tensors.h"

namespace Mki {
KernelBase::KernelBase(const std::string &opName, const BinHandle *handle) : kernelName_(opName), handle_(handle)
{
    if (handle_ != nullptr) {
        launchBufferSize_ = handle_->GetKernelTilingSize();
        int32_t coreType = handle_->GetKernelCoreType();
        switch (coreType) {
            case -1: MKI_LOG(ERROR) << "Failed to get core type!"; break;   // -1: get core type fail
            case 0: kernelType_ = KernelType::KERNEL_TYPE_AI_CORE; break;   // 0: AI_CORE
            case 2: kernelType_ = KernelType::KERNEL_TYPE_AIV; break;       // 2: AIV
            case 4: kernelType_ = KernelType::KERNEL_TYPE_MIX_AIC; break;   // 4: AIC
            default:
                MKI_LOG(WARN) << "Unexpected core type, use AIC as default!";
                kernelType_ = KernelType::KERNEL_TYPE_MIX_AIC;
        }
        MKI_LOG(DEBUG) << "Create kernel " << kernelName_ << ", launch buffer size " << launchBufferSize_
                       << ", coreType: " << kernelType_;
    } else {
        MKI_LOG(ERROR) << "Kernel " << kernelName_ << " handle is nullptr";
    }
}

KernelBase::~KernelBase() {}

std::string KernelBase::GetName() const { return kernelName_; }

KernelType KernelBase::GetType() const { return kernelType_; }

const BinHandle *KernelBase::GetBinHandle() const { return handle_; }

const KernelInfo &KernelBase::GetKernelInfo() const { return kernelInfo_; }

void KernelBase::SetLaunchWithTiling(bool flag) { kernelInfo_.SetLaunchWithTiling(flag); }

void KernelBase::SetTilingHostAddr(uint8_t *addr, uint64_t len) { kernelInfo_.SetTilingHostAddr(addr, len); }

void KernelBase::Reset()
{
    kernelInfo_.Reset();
}

Status KernelBase::Init(const LaunchParam &launchParam)
{
    MKI_CHECK(CheckInTensors(launchParam), "Not supported in tensors", return Status::FailStatus(1));
    MKI_CHECK(CanSupport(launchParam), "Not supported op", return Status::FailStatus(1));

    kernelInfo_.Reset();

    auto tilingSize = GetTilingSize(launchParam);
    bool launchWithTiling = kernelInfo_.GetLaunchWithTiling();
    if (launchWithTiling) {
        auto st = kernelInfo_.AllocTilingHost(tilingSize);
        MKI_CHECK(st.Ok(), "Failed to alloc host tiling buffer " << st.ToString(), return st);
    }

    auto status = InitImpl(launchParam);
    MKI_CHECK(status.Ok(), "Failed to init run info " << status.ToString(), return status);

    auto kernelParamNum = GetKernelParamNum(launchParam);
    uint64_t baseSize = kernelParamNum * sizeof(void *);
    uint64_t argsSize = baseSize;
    if (!launchWithTiling) {
        return kernelInfo_.InitArgs(argsSize);
    }

    uint64_t tilingUsedSize = Utils::GetTensorAlignedSize(kernelInfo_.GetTilingUsedSize());
    argsSize += tilingUsedSize;
    uint64_t constTensorSize = kernelInfo_.GetTilingSize() - kernelInfo_.GetConstTensorOffset();
    argsSize += constTensorSize;
    MKI_LOG(INFO) << "args num " << kernelParamNum << ", tiling used size " << tilingUsedSize
                  << ", const tensor size " << constTensorSize;
    status = kernelInfo_.InitArgs(argsSize);
    MKI_CHECK(status.Ok(), "failed to init args", return status);
    uint8_t *args = kernelInfo_.GetArgs();
    auto ret = memcpy_s(args + baseSize, argsSize - baseSize,
                        kernelInfo_.GetTilingHostAddr(), kernelInfo_.GetTilingUsedSize());
    MKI_CHECK(ret == EOK, "failed to copy tiling", return Status::FailStatus(-1));
    MKI_LOG(INFO) << "copy tiling data " << tilingUsedSize << " to args offset " << baseSize;
    if (constTensorSize > 0) {
        ret = memcpy_s(args + baseSize + tilingUsedSize, argsSize - baseSize - tilingUsedSize,
                       kernelInfo_.GetTilingHostAddr() + kernelInfo_.GetConstTensorOffset(), constTensorSize);
        MKI_CHECK(ret == EOK, "failed to copy const tensor", return Status::FailStatus(-1));
        MKI_LOG(INFO) << "copy const data " << constTensorSize << " to args offset " << baseSize + tilingUsedSize;
    }
    return Status::OkStatus();
}

Status KernelBase::Run(const LaunchParam &launchParam, RunInfo &runInfo)
{
    RtArgsExT argsEx;
    uint64_t argsNum = GetKernelParamNum(launchParam);
    uint8_t *argsPtr = kernelInfo_.GetArgs();
    uint64_t argsSize = kernelInfo_.GetArgsSize();
    MKI_CHECK(argsPtr != nullptr, "args size invalid", return Status::FailStatus(-1));
    MKI_CHECK(argsNum * sizeof(void *) <= argsSize, "args size invalid", return Status::FailStatus(-1));
    auto ret = memset_s(argsPtr, argsSize, 0, argsNum * sizeof(void *));
    MKI_CHECK(ret == EOK, "memory set failed", return Status::FailStatus(ERROR_INVALID_VALUE));
    void **args = reinterpret_cast<void **>(static_cast<void *>(argsPtr));
    // set hwsync
    auto status = UpdateHwsyncArgs(args, argsNum);
    MKI_CHECK(status.Ok(), "failed to update hwsync args", return status);
    // set const input
    size_t constTensorCount = kernelInfo_.GetConstTensorCount();
    RtHostInputInfoT hostInfo[constTensorCount];

    bool launchWithTiling = kernelInfo_.GetLaunchWithTiling();
    status = launchWithTiling ? UpdateConstTensorArgs(args, argsNum, hostInfo, constTensorCount)
                                        : UpdateConstTensorArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to update const tensor args", return status);
    // set input / output / workspace
    status = UpdateInOutWkspArgs(args, argsNum, launchParam, runInfo);
    MKI_CHECK(status.Ok(), "failed to update input output wksp args", return status);
    // set tiling
    status = launchWithTiling ? UpdateTilingArgs(argsEx, argsNum, runInfo)
                                        : UpdateTilingArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to get launch with tiling", return status);
    // Memset
    status = MemsetTensorArgs(args, argsNum, runInfo);
    MKI_CHECK(status.Ok(), "failed to memset tensor args", return status);
    // launch
    MkiRtKernelParam kernelParam{};
    kernelParam.tilingId = kernelInfo_.GetTilingId();
    kernelParam.blockDim = kernelInfo_.GetBlockDim();
    kernelParam.argsEx = &argsEx;
    argsEx.args = argsPtr;
    argsEx.argsSize = argsSize;
    if (launchWithTiling) {
        argsEx.hostInputInfoPtr = hostInfo;
        argsEx.hostInputInfoNum = constTensorCount;
    }
    MKI_LOG(INFO) << "Ready to run, KernelInfo:\n" << kernelInfo_.ToString();
    if (*handle_->GetHandle() != nullptr) {
        MKI_LOG(DEBUG) << "launch function with handle";
        int st = MkiRtFunctionLaunchWithHandle(*handle_->GetHandle(), &kernelParam, runInfo.GetStream(), nullptr);
        MKI_CHECK(
            st == MKIRT_SUCCESS, "Mki RtFunction LaunchWithHandle fail",
            return Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR, "Mki RtFunction LaunchWithHandle fail"));
    } else {
        MKI_LOG(DEBUG) << "launch function with flag";
        int st = MkiRtFunctionLaunchWithFlag(handle_->GetHandle(), &kernelParam, runInfo.GetStream(), nullptr);
        MKI_CHECK(st == MKIRT_SUCCESS, "Mki RtFunction LaunchWithFlag fail",
                    return Status::FailStatus(ERROR_LAUNCH_KERNEL_ERROR, "Mki RtFunction Launch fail"));
    }
    return Status::OkStatus();
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
    creator_ = other.creator_;
    kernelInfo_.Copy(other.kernelInfo_);
}

uint64_t KernelBase::GetTilingSize(const LaunchParam &launchParam) const
{
    UNUSED_VALUE(launchParam);
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
    uint64_t constInputNum = kernelInfo_.GetConstTensorCount();
    uint64_t workspaceNum = kernelInfo_.GetScratchSizes().size();
    uint64_t hwsyncNum = kernelInfo_.GetHwsyncIdx() < 0 ? 0 : 1;
    MKI_LOG(DEBUG) << "kernel param: " << inputOutputNum << " in/out, " << constInputNum << " const in, "
                  << workspaceNum << " workspaces, " << hwsyncNum << " hwsync";
    return inputOutputNum + constInputNum + workspaceNum + hwsyncNum + 1; // 1 is tiling
}

Status KernelBase::UpdateHwsyncArgs(void **args, uint64_t argsNum)
{
    int64_t hwsyncIdx = kernelInfo_.GetHwsyncIdx();
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
    uint64_t offset = Utils::GetTensorAlignedSize(kernelInfo_.GetTilingUsedSize()) + argsNum * sizeof(void *);
    for (uint64_t i = 0; i < infoNum; i++) {
        auto &constTensorInfo = kernelInfo_.GetConstTensorInfo(i);
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
    uint8_t *tilingDeviceAddr = runInfo.GetTilingDeviceAddr();
    uint64_t offset = kernelInfo_.GetConstTensorOffset();
    for (size_t i = 0; i < kernelInfo_.GetConstTensorCount(); i++) {
        auto &constTensorInfo = kernelInfo_.GetConstTensorInfo(i);
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
    auto &workspaces = kernelInfo_.GetScratchSizes();
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

Status KernelBase::UpdateTilingArgs(RtArgsExT &argsEx, uint64_t argsNum, const RunInfo &runInfo) const
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
    if (kernelInfo_.GetMemsetInfo().size() != 0) {
        Status status = ClearTensors(args, argsNum, kernelInfo_.GetMemsetInfo(), runInfo.GetStream());
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

void SetKernelSelfCreator(KernelBase &kernel, KernelBase::KernelSelfCreator func)
{
    MKI_CHECK(func != nullptr, "creator function is nullptr", return);
    kernel.creator_ = func;
}
} // namespace Mki
