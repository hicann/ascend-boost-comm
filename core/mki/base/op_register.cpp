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

#include "mki/base/op_register.h"
#include "mki/loader/device_kernel/device_kernel_loader.h"
#include "mki/utils/singleton/singleton.h"

namespace Mki {
HandleRegister::HandleRegister(const std::string &kerName) : kerName_(kerName) { RegisterBin(); }
KernelHandle HandleRegister::GetHandle() { return &handle_; }

void HandleRegister::RegisterBin()
{
    void *data = nullptr;
    uint32_t datalen = 0;
    int st = GetSingleton<KernelBinaryLoader>().GetBinaryDeviceCode(kerName_, data, datalen);
    if (st != MKIRT_SUCCESS) {
        MKI_LOG(WARN) << "No found device code for op " << kerName_;
        return;
    }

    size_t kernelNum = GetSingleton<KernelBinaryLoader>().GetBinaryKernelNum(kerName_);
    if (kernelNum == 0) {
        MKI_LOG(ERROR) << "Get Binary Kernel Num empty, op: " << kerName_;
        return;
    }

    uint32_t magic = GetSingleton<KernelBinaryLoader>().GetKernelMagic(kerName_);
    if (magic == 0) {
        MKI_LOG(ERROR) << "Get Magic empty, op: " << kerName_;
        return;
    }
    MkiRtModuleInfo moduleInfo;
    moduleInfo.type = MKIRT_MODULE_OBJECT;
    moduleInfo.version = 0;
    moduleInfo.data = data;
    moduleInfo.dataLen = datalen;
    moduleInfo.magic = magic;
    if (kernelNum == 1) {
        RegisterBinWithSingleKernel(moduleInfo);
    } else {
        RegisterBinWithMultiKernel(moduleInfo);
    }
}

void HandleRegister::RegisterBinWithSingleKernel(MkiRtModuleInfo &moduleInfo)
{
    MKI_LOG(DEBUG) << "SingleKernel RegisterBin start, opName:" << kerName_;
    int st = MkiRtModuleCreate(&moduleInfo, &moduleHandle_);
    if (st != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "Mki Create RtModule fail, error:" << st;
        return;
    }
    if (moduleHandle_ == nullptr) {
        MKI_LOG(ERROR) << "Mki Create RtModule fail, because it return null handle";
        return;
    }

    std::vector<std::string> kernelList;
    st = GetSingleton<KernelBinaryLoader>().GetBinaryKernelList(kerName_, kernelList);
    if (st != MKIRT_SUCCESS || kernelList.empty()) {
        MKI_LOG(ERROR) << "Mki Get TacticKernelList fail, error:" << st;
        return;
    }
    st = MkiRtModuleBindFunction(moduleHandle_, kernelList[0].c_str(), &handle_);
    if (st != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "Mki RtModuleGetFunction fail, errCode:" << st << ", errName:" << MkiRtErrorName(st)
                    << "errDesc:" << MkiRtErrorDesc(st);
    }
    MKI_LOG(DEBUG) << "SingleKernel RegisterBin finish, handle:" << handle_;
}

void HandleRegister::RegisterBinWithMultiKernel(MkiRtModuleInfo &moduleInfo)
{
    MKI_LOG(DEBUG) << "MultiKernel RegisterBin start, opName:" << kerName_;
    int st = AstRtRegisterAllFunction(&moduleInfo, &handle_);
    if (st != MKIRT_SUCCESS) {
        MKI_LOG(ERROR) << "Mki RtRegister AllFunction fail, error:" << st;
    }
    if (handle_ == nullptr) {
        MKI_LOG(ERROR) << "Mki RtRegister AllFunction fail, because it return null handle";
    }
    MKI_LOG(DEBUG) << "MultiKernel RegisterBin finish";
}

void SetKernelSelfCreator(KernelBase &kernel, KernelBase::KernelSelfCreator func)
{
    MKI_CHECK(func != nullptr, "creator function is nullptr", return);
    kernel.creator_ = func;
}
}
