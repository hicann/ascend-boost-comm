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
#ifndef MKI_UTILS_INIFILE_INI_FILE_H
#define MKI_UTILS_INIFILE_INI_FILE_H
#include <string>
#include <map>

namespace Mki {
class IniFile {
public:
    static bool ParseIniFileToMap(const std::string &iniFilePath,
        std::map<std::string, std::map<std::string, std::string>> &contentInfoMap, bool symCheck = true);
};
} // namespace Mki
#endif
