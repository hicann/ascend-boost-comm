/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "mki/utils/rt/base/base.h"
#include <map>
#include <string>
#include "mki/utils/rt/backend/backend_factory.h"

#define ERROR_ITEM(item)                                                                                               \
    {                                                                                                                  \
        item, "##item"                                                                                                 \
    }
namespace Mki {
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MIN = 0;
constexpr int VERSION_PATCH = 0;

void MkiRtGetVersion(int &major, int &min, int &patch)
{
    major = VERSION_MAJOR;
    min = VERSION_MIN;
    patch = VERSION_PATCH;
}

const char *MkiRtGetBuildInfo() { return "build time 2010222"; }

const char *MkiRtErrorName(int error)
{
    static std::map<int, std::string> errorNameMap = {
        ERROR_ITEM(MKIRT_ERROR_NOT_IMPLMENT),       ERROR_ITEM(MKIRT_ERROR_ASCEND_ENV_NOT_EXIST),
        ERROR_ITEM(MKIRT_ERROR_LOAD_RUNTIME_FAIL),  ERROR_ITEM(MKIRT_ERROR_FUNC_NOT_EXIST),
        ERROR_ITEM(MKIRT_ERROR_OPEN_BIN_FILE_FAIL), ERROR_ITEM(MKIRT_ERROR_PARA_CHECK_FAIL),
    };

    const auto it = errorNameMap.find(error);
    if (it != errorNameMap.cend()) {
        return it->second.c_str();
    }

    return "unknown";
}

const char *MkiRtErrorDesc(int error)
{
    static std::map<int, std::string> errorDescMap = {
        {MKIRT_ERROR_NOT_IMPLMENT, "not implment"},
        {MKIRT_ERROR_ASCEND_ENV_NOT_EXIST, "env ASCEND_HOME_PATH not exist"},
        {MKIRT_ERROR_LOAD_RUNTIME_FAIL, "load runtime library so"},
        {MKIRT_ERROR_FUNC_NOT_EXIST, "function not exist in runtime library"},
        {MKIRT_ERROR_OPEN_BIN_FILE_FAIL, "open bin file fail"},
        {MKIRT_ERROR_PARA_CHECK_FAIL, "para check fail"},
    };

    const auto it = errorDescMap.find(error);
    if (it != errorDescMap.cend()) {
        return it->second.c_str();
    }

    return "unknown";
}
}
