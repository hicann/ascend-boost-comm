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
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "op_desc_json.h"

namespace Mki {
namespace AutoGen {
void JsonToOpParam(const nlohmann::json &opDescJson, Mki::LaunchParam &launchParam)
{
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
}

Mki::Operation *GetOpByName(const std::string &operationName)
{
    MKI_LOG(ERROR) << "Please replace the libmki_test_autogen.so with your own version!";
    return nullptr;
}
} // namespace AutoGen
} // namespace Mki
