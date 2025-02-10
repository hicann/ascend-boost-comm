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
#ifndef MKI_UTILS_OPERATIONIR_OPERATION_IR_H
#define MKI_UTILS_OPERATIONIR_OPERATION_IR_H

#include <string>
#include <vector>
#include <map>
#include "mki/types.h"
#include "mki/utils/SVector/SVector.h"

namespace Mki {
// OperationIr中每个TensorInfoIr的supportedDtypes和supportedFormats的元素个数应该相等
struct TensorInfoIr {
    std::vector<TensorDType> supportedDtypes;
    std::vector<TensorFormat> supportedFormats;
    bool isOptional = false;
    std::string ToString() const;
};

class OperationIr {
public:
    OperationIr();
    ~OperationIr();
    std::string ToString();
    bool Parse(const std::map<std::string, std::string> &contentMap);
    const SVector<TensorInfoIr> &GetInTensorInfoIrs() const;
    const SVector<TensorInfoIr> &GetOutTensorInfoIrs() const;
    bool IsValid() const;
    size_t GetSupportSize() const;
    std::string GetCombString() const;
    bool ExtendInTensorIrByInputlens(SVector<int> &inputlens);

private:
    void ConvertStrToInt(const std::string &str, int &num) const;
    bool ParseKeyValue(const std::string &key, const std::string &value);
    bool ParseValueToInfoIrs(const std::string &tensorKey, SVector<TensorInfoIr> &tensorInfoIrs,
        const size_t &tensorIndex, const std::string &value) const;
    std::vector<TensorDType> GetParsedDTypes(const std::string &dTypeStrLine) const;
    std::vector<TensorFormat> GetParseFormats(const std::string &formatStrLine) const;

    bool SetSupportedDtypes(SVector<TensorInfoIr> &tensorInfoIrs, const size_t &index,
        const std::vector<TensorDType> &supportedDtypes) const;
    bool SetSupportedFormats(SVector<TensorInfoIr> &tensorInfoIrs, const size_t &index,
        const std::vector<TensorFormat> &supportedFormats) const;
    bool SetIsOptional(SVector<TensorInfoIr> &tensorInfoIrs, const size_t &index, const bool &isOptional) const;
    void InitIsValid();

private:
    SVector<TensorInfoIr> inTensorInfoIrs_;
    SVector<TensorInfoIr> outTensorInfoIrs_;
    bool isValid_ = false;
    size_t supportSize_ = 0;
};
} // namespace Mki
#endif
