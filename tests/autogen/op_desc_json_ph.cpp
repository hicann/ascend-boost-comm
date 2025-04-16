/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "op_desc_json.h"

namespace Mki {
namespace AutoGen {
void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)
{
    (void)opDescJson;
    (void)launchParam;
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
}

Mki::Operation *GetOpByName(const std::string &operationName)
{
    (void)operationName;
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
    return nullptr;
}
} // namespace AutoGen
} // namespace Mki
