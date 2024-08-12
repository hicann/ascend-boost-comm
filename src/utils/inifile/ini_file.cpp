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
#include "mki/utils/inifile/ini_file.h"
#include <cstring>
#include <fstream>
#include <algorithm>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/file_system/file_system.h"

namespace Mki {
constexpr uint32_t MAX_FILE_LINE_NUM = 20000;
bool IniFile::ParseIniFileToMap(const std::string &iniFilePath,
    std::map<std::string, std::map<std::string, std::string>> &contentInfoMap)
{
    std::string realPath = FileSystem::PathCheckAndRegular(iniFilePath);
    MKI_CHECK((FileSystem::FileSize(realPath) > 0 && FileSystem::FileSize(realPath) <= MAX_FILE_SIZE),
        "File size is invalid", return false);
    std::ifstream ifs(realPath);
    MKI_CHECK(ifs.is_open(), "Open ini file failed", return false);

    std::map<std::string, std::string> contentMap;
    contentMap.clear();
    contentInfoMap.clear();
    std::string line;
    std::string mapKey;
    uint32_t lineNum = 0;
    while (std::getline(ifs, line)) {
        if (lineNum > MAX_FILE_LINE_NUM) {
            MKI_LOG(ERROR) << "file lineNum is out of range, lineNum : " << lineNum;
            break;
        }
        lineNum++;
        // 去除空格
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (line.empty() || line.find('#') == 0) {
            continue;
        }
        if (line.find('[') == 0) {
            if (!mapKey.empty() && !contentMap.empty()) {
                contentInfoMap.emplace(make_pair(mapKey, contentMap));
                contentMap.clear();
            }
            size_t pos = line.rfind(']');
            if (pos == std::string::npos) {
                continue;
            }
            mapKey = line.substr(1, pos - 1);
            continue;
        }
        size_t posOfEqual = line.find('=');
        if (posOfEqual == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, posOfEqual);
        std::string value = line.substr(posOfEqual + 1, line.length() - posOfEqual - 1);
        if (!key.empty() && !value.empty()) {
            contentMap.emplace(make_pair(key, value));
        }
    }
    if (!contentMap.empty() && !mapKey.empty()) {
        contentInfoMap.emplace(make_pair(mapKey, contentMap));
        contentMap.clear();
    }
    ifs.close();
    return true;
}
}
