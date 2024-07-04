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
#ifndef CORE_BASE_KERNEL_REGISTER_H
#define CORE_BASE_KERNEL_REGISTER_H

#include <map>
#include <vector>

#include "kernel_base.h"
#include "mki/base/operation_base.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

namespace Mki {
using NewOperationFunc = Operation*(*)();
class OperationRegister {
public:
    OperationRegister(const char *opName, NewOperationFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        auto &operationCreators = OperationCreators();
        operationCreators.push_back(func);
        MKI_LOG(DEBUG) << "register operation " << opName;
    }

    static std::vector<NewOperationFunc> &OperationCreators()
    {
        static std::vector<NewOperationFunc> operationCreators;
        return operationCreators;
    }
};

class KernelRegister {
public:
    KernelRegister(const char *opName, const char *kerName, NewKernelFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        MKI_CHECK(kerName != nullptr, "kerName is nullptr", return);
        auto &kernelCreators = KernelCreators();
        kernelCreators[func] = opName;
        MKI_LOG(DEBUG) << "register kernel " << kerName << " of operation " << opName;
    }

    static std::unordered_map<NewKernelFunc, std::string> &KernelCreators()
    {
        static std::unordered_map<NewKernelFunc, std::string> kernelCreators;
        return kernelCreators;
    }
};

class HandleRegister {
public:
    HandleRegister(const std::string &kerName);
    KernelHandle GetHandle();

private:
    void RegisterBin();
    void RegisterBinWithSingleKernel(MkiRtModuleInfo &moduleInfo);
    void RegisterBinWithMultiKernel(MkiRtModuleInfo &moduleInfo);

private:
    std::string kerName_;
    void *handle_ = nullptr;
    void *moduleHandle_ = nullptr;
};

struct BinaryBasicInfo {
    const uint8_t *binaryBuf = nullptr;
    uint32_t binaryLen = 0;
    std::string targetSoc;
};

class KernelBinaryRegister {
public:
    KernelBinaryRegister(const char *soc, const char *opName, const uint8_t *binary, uint32_t binaryLen) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        MKI_CHECK(soc != nullptr, "target soc is nullptr", return);
        MKI_CHECK(binary != nullptr, "binary addr is nullptr", return);
        auto &binaryMap = GetKernelBinaryMap();
        binaryMap[std::string(opName)].push_back({ binary, binaryLen, soc });
        MKI_LOG(DEBUG) << "register kernel binary " << opName << "-" << soc << " success";
    }
    static std::map<std::string, std::vector<BinaryBasicInfo>> &GetKernelBinaryMap()
    {
        static std::map<std::string, std::vector<BinaryBasicInfo>> binaryMap;
        return binaryMap;
    }
};

void SetKernelSelfCreator(KernelBase &kernel, KernelBase::KernelSelfCreator func);

#define REG_OPERATION(opName)                                                                                          \
    Operation *GetOperation##opName()                                                                                  \
    {                                                                                                                  \
        static opName op##opName(#opName);                                                                             \
        return &op##opName;                                                                                            \
    }                                                                                                                  \
    static OperationRegister op##opName##register = OperationRegister(#opName, GetOperation##opName)

#define REG_KERNEL_BASE(kerName)                                                                                       \
    Kernel const *GetKernel##kerName()                                                                                 \
    {                                                                                                                  \
        static HandleRegister handleRegister##kerName(#kerName);                                                       \
        static kerName ker##kerName(#kerName, handleRegister##kerName.GetHandle());                                    \
        SetKernelSelfCreator(ker##kerName, []() \
        { return new kerName(#kerName, handleRegister##kerName.GetHandle()); });                                       \
        return &ker##kerName;                                                                                          \
    }                                                                                                                  \
    static KernelRegister ker##kerName##register = KernelRegister(OperationPlaceHolder, #kerName, GetKernel##kerName)

#define REG_KERNEL(soc, kerName, binary)                                                                               \
    static KernelBinaryRegister bin##kerName##soc##register = \
        KernelBinaryRegister(#soc, #kerName, binary, sizeof(binary))
} // namespace Mki

#endif
