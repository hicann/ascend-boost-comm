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
#ifndef MKI_OPERATION_IR_CFG_H
#define MKI_OPERATION_IR_CFG_H

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
