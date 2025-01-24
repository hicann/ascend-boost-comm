/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MKI_UTILS_CFG_CFG_ITEM_H
#define MKI_UTILS_CFG_CFG_ITEM_H

#include <string>

namespace Mki {
struct LogCfg {
    bool isLogToStdOut = false;
    bool isLogToFile = false;
    bool isLogToFileFlush = false;
    std::string logLevel = "INFO";
    std::string logToBoostType = "atb";
    std::string logPath = "~";
};
} // namespace Mki
#endif
