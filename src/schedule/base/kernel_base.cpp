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
#include "mki/utils/rt/rt.h"
#include "mki/utils/math/tensor_utils.h"
#include "mki/utils/memset/clear_tensors.h"

namespace Mki {
class KernelParamBuilder {
public:
    Status Init(const LaunchParam &launchParam, const RunInfo &runInfo, uint64_t argsNum, const KernelInfo &kernelInfo)
    {
        uint8_t *argsPtr = kernelInfo.GetArgs();
        uint64_t argsSize = kernelInfo.GetArgsSize();
        MKI_CHECK(argsPtr != nullptr, "args size invalid", return Status::FailStatus(-1));
        MKI_CHECK(argsNum * sizeof(void *) <= argsSize, "args size invalid", return Status::FailStatus(-1));
        auto ret = memset_s(argsPtr, argsSize, 0, argsNum * sizeof(void *));
        MKI_CHECK(ret == EOK, "memory set failed", return Status::FailStatus(ERROR_INVALID_VALUE));
        void **args = reinterpret_cast<void **>(static_cast<void *>(argsPtr));
        // set hwsync
        int64_t hwsyncIdx = kernelInfo.GetHwsyncIdx();
        auto status = UpdateHwsyncArgs(args, argsNum, hwsyncIdx);
        MKI_CHECK(status.Ok(), "failed to update hwsync args", return status);
        // set const input
        bool launchWithTiling = kernelInfo.GetLaunchWithTiling();
        const auto &constTensorInfos = kernelInfo.GetConstTensorInfos();
        // size_t hostInputCount = 
        if (launchWithTiling) {
            constexpr size_t maxConstTensorCount = 1024;
            size_t constTensorCount = kernelInfo.GetConstTensorCount();
            if (constTensorCount > 0) {
                MKI_CHECK(constTensorCount < maxConstTensorCount, "const tensor size check failed, is "
                          << constTensorCount, return Status::FailStatus(-1));
                hostInfo_.reset(new (std::nothrow) RtHostInputInfoT[constTensorCount]);
                MKI_CHECK(hostInfo_ != nullptr, "hostInfo size nullptr", return Status::FailStatus(-1));
                uint64_t constTensorOffset =
                    Utils::GetTensorAlignedSize(kernelInfo.GetTilingUsedSize()) + argsNum * sizeof(void *);
                status = UpdateConstTensorArgs(args, hostInfo_.get(), constTensorOffset, constTensorInfos);
                MKI_CHECK(status.Ok(), "failed to update const tensor args, tiling flag: " << launchWithTiling,
                          return status);
            }
            argsEx_.hostInputInfoPtr = hostInfo_.get();
            argsEx_.hostInputInfoNum = constTensorCount;
        } else {
            uint64_t constTensorOffset = kernelInfo.GetConstTensorOffset();
            status = UpdateConstTensorArgs(args, runInfo.GetTilingDeviceAddr(), constTensorOffset, constTensorInfos);
            MKI_CHECK(status.Ok(), "failed to update const tensor args, tiling flag: " << launchWithTiling,
                      return status);
        }

        bool launchWithTensorlist = kernelInfo.GetLaunchWithTensorlist();
        const auto &workspaces = kernelInfo.GetScratchSizes();
        if (launchWithTensorlist) {
            status = UpdateArgsWithTensorList(args, argsNum, workspaces,
                        launchParam, runInfo.GetScratchDeviceAddr());
            MKI_CHECK(status.Ok(), "failed to update args with tensorlist: " << launchWithTensorlist,
                        return status);
        } else {
            // set input / output / workspace
            status = UpdateInOutWkspArgs(args, argsNum, workspaces, launchParam, runInfo.GetScratchDeviceAddr());
            MKI_CHECK(status.Ok(), "failed to update input output wksp args", return status);
        }
        // set tiling
        status = launchWithTiling ? UpdateTilingArgs(argsEx_, argsNum)
                                  : UpdateTilingArgs(args, argsNum, runInfo.GetTilingDeviceAddr());
        MKI_CHECK(status.Ok(), "failed to get launch with tiling", return status);
        // Memset
        const auto &memsetInfo = kernelInfo.GetMemsetInfo();
        status = MemsetTensorArgs(args, argsNum, runInfo.GetStream(), memsetInfo);
        MKI_CHECK(status.Ok(), "failed to memset tensor args", return status);
        // launch
        kernelParam_.tilingId = kernelInfo.GetTilingId();
        kernelParam_.blockDim = kernelInfo.GetBlockDim();
        kernelParam_.argsEx = &argsEx_;
        argsEx_.args = argsPtr;
        argsEx_.argsSize = argsSize;
        return status;
    }

    const MkiRtKernelParam &GetKernelParam() const
    {
        return kernelParam_;
    }

private:
    Status UpdateHwsyncArgs(void **args, uint64_t argsNum, int64_t hwsyncIdx) const
    {
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

    Status UpdateInOutWkspArgs(void **args, uint64_t argsNum, const MiniVector<uint64_t> &workspaces,
                               const LaunchParam &launchParam, uint8_t *workspaceAddr) const
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

    Status MemsetTensorArgs(void **args, uint64_t argsNum, void *stream,
                            const MiniVector<KernelInfo::MemsetInfo> &memsetInfo) const
    {
        if (memsetInfo.size() != 0) {
            Status status = ClearTensors(args, argsNum, memsetInfo, stream);
            MKI_CHECK(status.Ok(), "failed to clear tensors", return status);
        }
        return Status::OkStatus();
    }

    Status UpdateConstTensorArgs(void **args, RtHostInputInfoT *info, uint64_t constTensorOffset,
                                 const MiniVector<KernelInfo::ConstTensorInfo> &constTensorInfos) const
    {
        uint64_t offset = constTensorOffset;
        for (uint64_t i = 0; i < constTensorInfos.size(); i++) {
            auto &constTensorInfo = constTensorInfos.at(i);
            args[constTensorInfo.argIdx] = info + i; // placeholder
            info[i].addrOffset = constTensorInfo.argIdx * sizeof(void *);
            info[i].dataOffset = offset;
            MKI_LOG(DEBUG) << "args info: const tensor " << constTensorInfo.argIdx << " offset in args " << offset;
            offset += constTensorInfo.size;
        }
        return Status::OkStatus();
    }

    Status UpdateConstTensorArgs(void **args, uint8_t *tilingDeviceAddr, uint64_t constTensorOffset,
                                 const MiniVector<KernelInfo::ConstTensorInfo> &constTensorInfos) const
    {
        uint64_t offset = constTensorOffset;
        for (size_t i = 0; i < constTensorInfos.size(); i++) {
            auto &constTensorInfo = constTensorInfos.at(i);
            args[constTensorInfo.argIdx] = tilingDeviceAddr + offset;
            MKI_LOG(DEBUG) << "args info: const tensor " << constTensorInfo.argIdx << " offset in tiling " << offset;
            offset += constTensorInfo.size;
        }
        return Status::OkStatus();
    }

    Status UpdateTilingArgs(RtArgsExT &argsEx, uint64_t argsNum) const
    {
        MKI_CHECK((argsNum > 1), "argsNum invalid : " << argsNum, return Status::FailStatus(-1));
        MKI_LOG(DEBUG) << "argsNum: tiling " << (argsNum - 1);
        argsEx.hasTiling = 1;
        argsEx.tilingAddrOffset = (argsNum - 1) * sizeof(void *);
        argsEx.tilingDataOffset = argsNum * sizeof(void *);
        return Status::OkStatus();
    }

    Status UpdateTilingArgs(void **args, uint64_t argsNum, uint8_t *tilingDeviceAddr) const
    {
        MKI_LOG(DEBUG) << "args info: tiling " << (argsNum - 1);
        args[argsNum - 1] = tilingDeviceAddr;
        return Status::OkStatus();
    }

    Status UpdateArgsWithTensorList(void **args, uint64_t argsNum, const MiniVector<uint64_t> &workspaces,
                            const LaunchParam &launchParam, uint8_t *workspaceAddr)
    {
        size_t inputNum = launchParam.GetInputLenCount() > 0 ? launchParam.GetInputLenCount() : launchParam.GetInTensorCount();
        size_t outputNum = launchParam.GetOutputLenCount() > 0 ? launchParam.GetOutputLenCount() : launchParam.GetOutTensorCount();
        SVector<int> inputLens = launchParam.GetInputLens();
        size_t tensorId = 0;
        for (size_t i = 0 ; i < inputNum ; i++) {
            int len = launchParam.GetInputLenCount() > 0 ? inputLens[i] : 1;
            if (len > 1) {
                tensorId += len;
            } else if (len == 1) {
                MKI_LOG(DEBUG) << "args info: input " << i << " tensorId: " << tensorId;
                args[i] = launchParam.GetInTensor(tensorId++).data;
            }
        }
        SVector<int> outputLens = launchParam.GetOutputLens();
        tensorId = 0;
        for (size_t i = 0 ; i < outputNum ; i++) {
            int len = launchParam.GetOutputLenCount() > 0 ? outputLens[i] : 1;
            if (len > 1) {
                tensorId += len;
            } else if (len == 1) {
                MKI_LOG(DEBUG) << "args info: output " << i + inputNum << " tensorId: " << tensorId;
                args[i + inputNum] = launchParam.GetOutTensor(tensorId++).data;
            }
        }
        size_t workspaceNum = workspaces.size();
        uint64_t workspaceOffset = 0;
        uint64_t idx = inputNum + outputNum;
        for (size_t i = 0; i < workspaceNum && idx < argsNum; idx++) {
            if (args[idx] != nullptr) {
                continue;
            }
            MKI_LOG(DEBUG) << "args info: workspace " << idx << " offset " << workspaceOffset;
            args[idx] = workspaceAddr + workspaceOffset;
            workspaceOffset += workspaces.at(i++);
        }
        return Status::OkStatus();
    }

private:
    RtArgsExT argsEx_;
    std::unique_ptr<RtHostInputInfoT[]> hostInfo_{nullptr};
    MkiRtKernelParam kernelParam_;
};

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

    if (launchParam.GetInputLenCount() > 0 || launchParam.GetOutputLenCount()) {
        kernelInfo_.SetLaunchWithTensorlist(true);
    }
    bool launchWithTensorlist = kernelInfo_.GetLaunchWithTensorlist();
    auto tensorListSize = Utils::GetTensorAlignedSize(GetTensorListSize(launchParam));
    if (launchWithTensorlist) {
        auto st = kernelInfo_.AllocTensorListHost(tensorListSize);
    }
    auto status = InitImpl(launchParam);
    MKI_CHECK(status.Ok(), "Failed to init run info " << status.ToString(), return status);
    status = InitTensorList(launchParam);
    MKI_CHECK(status.Ok(), "Failed to init tensorList info " << status.ToString(), return status);

    auto kernelParamNum = GetKernelArgsNum(launchParam);
    uint64_t baseSize = kernelParamNum * sizeof(void *);
    uint64_t argsSize = baseSize;
    if (!launchWithTiling) {
        return kernelInfo_.InitArgs(argsSize);
    }

    uint64_t tilingUsedSize = Utils::GetTensorAlignedSize(kernelInfo_.GetTilingUsedSize());
    argsSize += tilingUsedSize;
    uint64_t constTensorSize = kernelInfo_.GetTilingSize() - kernelInfo_.GetConstTensorOffset();
    argsSize += constTensorSize;
    argsSize += tensorListSize;
    MKI_LOG(INFO) << "args num " << kernelParamNum << ", tiling used size " << tilingUsedSize
                  << ", const tensor size " << constTensorSize << ", tensorList size " << tensorListSize;
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
    if (tensorListSize > 0) {
        ret = memcpy_s(args + baseSize + tilingUsedSize + constTensorSize, argsSize - baseSize - tilingUsedSize - constTensorSize,
                       kernelInfo_.GetTensorListHostAddr(), tensorListSize);
        MKI_CHECK(ret == EOK, "failed to copy tensorList", return Status::FailStatus(-1));
        MKI_LOG(INFO) << "copy tensorList data " << tensorListSize << " to args offset " << baseSize + kernelInfo_.GetTilingSize();
    }
    return Status::OkStatus();
}

uint64_t KernelBase::GetKernelArgsNum(const LaunchParam &launchParam)
{
    size_t inputNum = launchParam.GetInputLenCount() > 0 ? launchParam.GetInputLenCount() : launchParam.GetInTensorCount();
    size_t outputNum = launchParam.GetOutputLenCount() > 0 ? launchParam.GetOutputLenCount() : launchParam.GetOutTensorCount();
    uint64_t inputOutputNum = inputNum + outputNum;
    uint64_t constInputNum = kernelInfo_.GetConstTensorCount();
    uint64_t workspaceNum = kernelInfo_.GetScratchSizes().size();
    uint64_t hwsyncNum = kernelInfo_.GetHwsyncIdx() < 0 ? 0 : 1;
    MKI_LOG(DEBUG) << "kernel param: " << inputOutputNum << " in/out, " << constInputNum << " const in, "
                << workspaceNum << " workspaces, " << hwsyncNum << " hwsync";
    return inputOutputNum + constInputNum + workspaceNum + hwsyncNum + 1; // 1 is tiling
}

Status KernelBase::Run(const LaunchParam &launchParam, RunInfo &runInfo)
{
    KernelParamBuilder paramBuilder;
    uint64_t argsNum = GetKernelArgsNum(launchParam);
    Status status = paramBuilder.Init(launchParam, runInfo, argsNum, kernelInfo_);
    MKI_CHECK(status.Ok(), "failed to build kernel params", return status);
    const MkiRtKernelParam &kernelParam = paramBuilder.GetKernelParam();
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

uint64_t KernelBase::GetTensorListSize(const LaunchParam &launchParam)
{
    size_t tensorId = 0;
    uint64_t totalSize = 0;
    for (auto &len : launchParam.GetInputLens()) {
        if (len > 1) {
            uint64_t sigleSize = len + 1;
            for (size_t i = tensorId; i < tensorId + len; ++i) {
                sigleSize += launchParam.GetInTensors().at(i).desc.dims.size() + 1;
            }
            totalSize += sigleSize;
            tensorId += len;
        } else {
            ++tensorId;
        }
    }
    tensorId = 0;
    for (auto &len : launchParam.GetOutputLens()) {
        if (len > 1) {
            uint64_t sigleSize = len + 1;
            for (size_t i = tensorId; i < tensorId + len; ++i) {
                sigleSize += launchParam.GetOutTensors().at(i).desc.dims.size() + 1;
            }
            totalSize += sigleSize;
            tensorId += len;
        } else {
            ++tensorId;
        }
    }
    return totalSize * sizeof(uint64_t);
}

void KernelBase::BuildTensorList(uint8_t *startPtr, SVector<int> &lens, SVector<Tensor> &tensors, uint64_t &useSize, uint64_t idxOffset)
{
    size_t tensorId = 0;
    uint64_t totalSize = 0;
    uint64_t *basePtr = reinterpret_cast<uint64_t *>(startPtr);
    for (uint64_t i = 0 ; i < lens.size() ; i++) {
        int len = lens[i];
        if (len > 1) {
            uint64_t sigleSize = len + 1;
            for (size_t i = tensorId; i < tensorId + len; ++i) {
                sigleSize += tensors.at(i).desc.dims.size() + 1;
            }
            uint64_t dataOffset = sigleSize - len;
            uint64_t curOffset = 1;
            *basePtr = dataOffset * 8;
            for (size_t i = tensorId; i < tensorId + len; ++i) {
                *reinterpret_cast<uint32_t *>(basePtr + curOffset) = tensors.at(i).desc.dims.size();
                *(reinterpret_cast<uint32_t *>(basePtr + curOffset) + 1) = i - tensorId;
                curOffset++;
                for (size_t j = 0; j < tensors.at(i).desc.dims.size(); ++j) {
                    *(basePtr + curOffset++) = tensors.at(i).desc.dims[j];
                }
                *(basePtr + dataOffset + i - tensorId) = reinterpret_cast<intptr_t>(tensors.at(i).data);
            }
            tensorId += len;
            basePtr = basePtr + sigleSize;
            totalSize += sigleSize;
            kernelInfo_.AddTensorListInfo(i + idxOffset, sigleSize * sizeof(uint64_t));
        } else if (len == 1) {
            ++tensorId;
        }
    }
    useSize = totalSize * sizeof(uint64_t);
}

Status KernelBase::InitTensorList(const LaunchParam &launchParam)
{
    uint8_t *startPtr = kernelInfo_.GetTensorListHostAddr();
    uint64_t useSize = 0;
    SVector<Tensor> inTensors = launchParam.GetInTensors();
    SVector<int> inputLens = launchParam.GetInputLens();
    BuildTensorList(startPtr, inputLens, inTensors, useSize, 0);
    startPtr += useSize;
    SVector<Tensor> outTensors = launchParam.GetOutTensors();
    SVector<int> outputLens = launchParam.GetOutputLens();
    BuildTensorList(startPtr, outputLens, outTensors, useSize, inputLens.size());
    return Status::OkStatus();
}

Status KernelBase::InitImpl(const LaunchParam &launchParam)
{
    UNUSED_VALUE(launchParam);
    MKI_CHECK(false, "InitImpl has to be overriden!", return Status::FailStatus(-1));
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
