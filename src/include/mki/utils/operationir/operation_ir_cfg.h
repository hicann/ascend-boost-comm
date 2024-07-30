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
#ifndef MKI_UTILS_OPERATIONIR_OPERATION_IR_CFG_H
#define MKI_UTILS_OPERATIONIR_OPERATION_IR_CFG_H

#include <map>
#include "operation_ir.h"
#include "mki/utils/status/status.h"

namespace Mki {
class OperationIrCfg {
public:
    OperationIrCfg();
    ~OperationIrCfg();
    Status Load(const std::string &fileName);
    OperationIr *GetOperationIr(const std::string &opKey);

private:
    bool IsValidOpKey(const std::string &key) const;

private:
    std::map<std::string, OperationIr> data_;
};
} // namespace Mki
#endif
