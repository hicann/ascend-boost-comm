/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MKI_UTILS_CFG_CFG_CORE_H
#define MKI_UTILS_CFG_CFG_CORE_H

#include <string>
#include <map>
#include "mki/utils/status/status.h"
#include "mki/types.h"
#include "cfg_item.h"

namespace Mki {
class CfgCore {
public:
    CfgCore();
    virtual ~CfgCore() = default;
    static CfgCore &GetCfgCoreInstance();
    bool CfgFileExists() const;
    const LogCfg GetLogCfg() const;
    Status LoadFromFile(const std::string &fileName);
    Status ParseLogData();

private:
    void InitMkiHomePath();
    std::map<std::string, std::map<std::string, std::string>> rawInfoMap_;
    LogCfg logCfgData_;
    bool cfgFileExixts_ = false;
    bool printLog_ = false;
    std::string mkiHomePath_;
};
} // namespace Mki
#endif
