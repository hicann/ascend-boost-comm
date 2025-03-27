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
#include "mki/utils/operationir/operation_ir_cfg.h"
#include <fstream>
#include <algorithm>
#include <string>
#include "mki/types.h"
#include "mki/utils/log/log.h"
#include "mki/utils/assert/assert.h"
#include "mki/utils/file_system/file_system.h"
#include "mki/utils/inifile/ini_file.h"

namespace Mki {
constexpr size_t MAX_OP_NAME_LEN = 128;

OperationIrCfg::OperationIrCfg() {}

OperationIrCfg::~OperationIrCfg() {}

Status OperationIrCfg::Load(const std::string &fileName)
{
    std::map<std::string, std::map<std::string, std::string>> contentInfoMap;
    bool isParseSuccess = IniFile::ParseIniFileToMap(fileName, contentInfoMap);
    MKI_CHECK(isParseSuccess && !contentInfoMap.empty(), "ParseIniFileToMap failed!",
        return Status::FailStatus(ERROR_INVALID_VALUE));
    std::string currentOpKey;
    OperationIr opIr;
    std::map<std::string, std::map<std::string, std::string>>::iterator it;
    for (it = contentInfoMap.begin(); it != contentInfoMap.end(); it++) {
        currentOpKey = it->first;
        MKI_CHECK(IsValidOpKey(currentOpKey), "OpKey is invalid! key:" << currentOpKey,
            return Status::FailStatus(ERROR_INVALID_VALUE));
        opIr = OperationIr();
        isParseSuccess = opIr.Parse(it->second);

        MKI_CHECK(isParseSuccess, "parse failed!", return Status::FailStatus(ERROR_INVALID_VALUE));
        data_.emplace(make_pair(currentOpKey, opIr));
    }
    return Status::OkStatus();
}

bool OperationIrCfg::IsValidOpKey(const std::string &key) const
{
    MKI_CHECK(key.length() > 0 && key.length() <= MAX_OP_NAME_LEN,
        "The length of OpKey must be greater than 0 and less than or equal to 128.", return false);
    for (const char &c : key) {
        MKI_CHECK(isalnum(c), "opName only characters and digits are supported.", return false);
    }
    return true;
}

OperationIr *OperationIrCfg::GetOperationIr(const std::string &opKey)
{
    auto it = data_.find(opKey);
    if (it != data_.end()) {
        MKI_LOG(DEBUG) << opKey << ":\n" << it->second.ToString();
        return &it->second;
    }
    return nullptr;
}
OperationIrTensorlistHandle::OperationIrTensorlistHandle()
{
    this->copyPool.clear();
}

OperationIrTensorlistHandle::~OperationIrTensorlistHandle()
{
    for (uint64_t i = 0; i < this->copyPool.size(); i++) {
        if (this->copyPool[i] != nullptr) {
            delete this->copyPool[i];
            this->copyPool[i] = nullptr;
        }
    }
    this->copyPool.clear();
}

Mki::OperationIr* OperationIrTensorlistHandle::ExtendInTensorIrByInputlens(Mki::SVector<int> &inputlens, Mki::OperationIr* outIr)
{
    if (outIr == nullptr) {
        MKI_LOG(ERROR) << "The InTensorIr is empty";
        return nullptr;
    }
    Mki::OperationIr* temp = new Mki::OperationIr;
    std::copy(outIr, outIr + 1, temp);
    this->copyPool.push_back(temp);
    temp->ExtendInTensorIrByInputlens(inputlens);
    return temp;
}

Mki::OperationIr* OperationIrTensorlistHandle::ExtendOutTensorIrByOutputlens(Mki::SVector<int> &outputlens, Mki::OperationIr* outIr)
{
    if (outIr == nullptr) {
        MKI_LOG(ERROR) << "The OutTensorIr is empty";
        return nullptr;
    }
    Mki::OperationIr* temp = new Mki::OperationIr;
    std::copy(outIr, outIr + 1, temp);
    this->copyPool.push_back(temp);
    temp->ExtendOutTensorIrByOutputlens(outputlens);
    return temp;
}

Mki::OperationIr* OperationIrTensorlistHandle::ExtendInAndOutTensorIrByInputlens(Mki::SVector<int> &inputlens, Mki::SVector<int> &outputlens, Mki::OperationIr* outIr)
{
    if (outIr == nullptr) {
        MKI_LOG(ERROR) << "The OutAndInTensorIr is empty";
        return nullptr;
    }
    Mki::OperationIr* temp = new Mki::OperationIr;
    std::copy(outIr, outIr + 1, temp);
    this->copyPool.push_back(temp);
    temp->ExtendInTensorIrByInputlens(inputlens);
    temp->ExtendOutTensorIrByOutputlens(outputlens);
    return temp;
}
} // namespace Mki
