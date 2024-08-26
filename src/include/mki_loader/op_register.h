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
#ifndef MKI_LOADER_OP_REGISTER_H
#define MKI_LOADER_OP_REGISTER_H

#include <map>
#include <vector>
#include "mki/base/kernel_base.h"
#include "mki/base/operation_base.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"

#define MACRO_TO_STR(name) #name
#define MKI_NAMESPACE_LOG(level, spaceName) MKI_LOG(level) << (MACRO_TO_STR(spaceName))

namespace OpSpace {
using NewOperationFunc = Mki::Operation*(*)();
using NewKernelFunc = const Mki::Kernel*(*)(const Mki::BinHandle *);

class OperationRegister {
public:
    OperationRegister(const char *opName, NewOperationFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        auto &operationCreators = GetOperationCreators();
        operationCreators.push_back(func);
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register operation " << opName;
    }

    static std::vector<NewOperationFunc> &GetOperationCreators()
    {
        static std::vector<NewOperationFunc> operationCreators;
        return operationCreators;
    }
};

class KernelRegister {
public:
    struct CreatorInfo {
        NewKernelFunc func;
        std::string opName;
        std::string kernelName;
    };

    KernelRegister(const char *opName, const char *kerName, NewKernelFunc func) noexcept
    {
        MKI_CHECK(opName != nullptr, "opName is nullptr", return);
        MKI_CHECK(kerName != nullptr, "kerName is nullptr", return);
        auto &kernelCreators = GetKernelCreators();
        kernelCreators.push_back({ func, opName, kerName });
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register kernel " << kerName << " of operation " << opName;
    }

    static std::vector<CreatorInfo> &GetKernelCreators()
    {
        static std::vector<CreatorInfo> kernelCreators;
        return kernelCreators;
    }
};

class KernelBinaryRegister {
public:
    KernelBinaryRegister(const char *soc, const char *kernelName, const uint8_t *binary, uint32_t binaryLen) noexcept
    {
        MKI_CHECK(kernelName != nullptr, "kernelName is nullptr", return);
        MKI_CHECK(soc != nullptr, "target soc is nullptr", return);
        MKI_CHECK(binary != nullptr, "binary addr is nullptr", return);
        auto &binaryMap = GetKernelBinaryMap();
        binaryMap[std::string(kernelName)].push_back({ binary, binaryLen, soc });
        MKI_NAMESPACE_LOG(DEBUG, OpSpace) << " register kernel binary " << kernelName << "-" << soc << " success";
    }

    static std::map<std::string, std::vector<Mki::BinaryBasicInfo>> &GetKernelBinaryMap()
    {
        static std::map<std::string, std::vector<Mki::BinaryBasicInfo>> binaryMap;
        return binaryMap;
    }
};

#define REG_OPERATION(opName)                                                                                      \
    Mki::Operation *GetOperation##opName()                                                                         \
    {                                                                                                              \
        static opName op##opName(#opName);                                                                         \
        return &op##opName;                                                                                        \
    }                                                                                                              \
    static OperationRegister op##opName##register = OperationRegister(#opName, GetOperation##opName)

#define REG_KERNEL_BASE(kerName)                                                                                   \
    Mki::Kernel const *GetKernel##kerName(const Mki::BinHandle *binHandle)                                         \
    {                                                                                                              \
        static kerName ker##kerName(#kerName, binHandle);                                                          \
        SetKernelSelfCreator(ker##kerName, [=](){ return new kerName(#kerName, binHandle); });                     \
        return &ker##kerName;                                                                                      \
    }                                                                                                              \
    static KernelRegister ker##kerName##register = KernelRegister(OperationPlaceHolder, #kerName, GetKernel##kerName)

#define REG_KERNEL(soc, kerName, binary)                                                                           \
    static KernelBinaryRegister bin##kerName##soc##register =                                                      \
           KernelBinaryRegister(#soc, #kerName, binary, sizeof(binary))
} // namespace OpSpace

#endif
