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
#ifndef MKI_OPERATION_IR_H
#define MKI_OPERATION_IR_H

#include <string>
#include <vector>
#include <map>
#include "mki/utils/svector/svector.h"

namespace Mki {
// OperationIr中每个TensorInfoIr的supportedDtypes和supportedFormats的元素个数应该相等
struct TensorInfoIr {
    std::vector<int> supportedDtypes;
    std::vector<int> supportedFormats;
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

private:
    void ConvertStrToInt(const std::string &str, int &num) const;
    bool ParseKeyValue(const std::string &key, const std::string &value);
    bool ParseValueToInfoIrs(const std::string &tensorKey, SVector<TensorInfoIr> &tensorInfoIrs,
        const size_t &tensorIndex, const std::string &value) const;
    std::vector<int> GetParsedDTypes(const std::string &dTypeStrLine) const;
    std::vector<int> GetParseFormats(const std::string &formatStrLine) const;

    bool SetSupportedDtypes(SVector<TensorInfoIr> &tensorInfoIrs, const size_t &index,
        const std::vector<int> &supportedDtypes) const;
    bool SetSupportedFormats(SVector<TensorInfoIr> &tensorInfoIrs, const size_t &index,
        const std::vector<int> &supportedFormats) const;
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
