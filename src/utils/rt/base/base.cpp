/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
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
