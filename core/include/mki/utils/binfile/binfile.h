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
#ifndef MKI_BINFILE_BINFILE_H
#define MKI_BINFILE_BINFILE_H
#include <string>
#include <cstdint>
#include <fstream>
#include <set>
#include <vector>
#include <sys/stat.h>
#include "mki/utils/status/status.h"

namespace Mki {
const std::string ATTR_VERSION = "$Version";
const std::string ATTR_END = "$End";
const std::string ATTR_OBJECT_LENGTH = "$Object.Length";
const std::string ATTR_OBJECT_COUNT = "$Object.Count";
const std::string ATTR_OBJECT_PREFIX = "$Object.";
constexpr mode_t BIN_FILE_MODE = S_IRUSR | S_IWUSR | S_IRGRP;
/**
 Composition of BinFile:
 $Version=xx
 $Object.Count=xx
 $Object.Length=xx
 Attr1=xx
 Attr2=xx
 ...
 AttrN=xx
 $Object.XxxTactic=offset(from 0),length
 $Object.YyyTactic=offset,length
 ...
 $Object.ZzzTactic=offset,length
 $End=1
 BinaryData of XxxTactic
 BinaryData of YyyTactic
 ...
 BinaryData of ZzzTactic

 ps.Head is frome $Version to $End
*/
class BinFile {
struct Binary {
    uint64_t offset = 0;
    uint64_t length = 0;
};

public:
    BinFile();
    ~BinFile();

    Status AddAttr(const std::string &name, const std::string &value);
    Status AddObject(const std::string &name, void *binaryBuffer, uint64_t binaryLen);

    Status Write(const std::string &filePath, const mode_t mode = BIN_FILE_MODE);
    Status Read(const std::string &filePath);

    void GetAllAttrs(std::vector<std::pair<std::string, std::string>> &attrs);
    void GetAllObjects(std::vector<std::pair<std::string, std::pair<void *, uint64_t>>> &binaries);

private:
    bool WriteImpl(int &fd);
    bool WriteAttr(int &fd, const std::string &name, const std::string &value) const;
    Status ParseBinFile(std::ifstream &fd, size_t fileSize);
    Status ParseSystemAttr(const std::string &attrName, const std::string &value);

private:
    std::string version_ = "1.0";

    std::set<std::string> attrNames_;
    std::vector<std::pair<std::string, std::string>> attrs_; // <attrName, attrValue>

    std::set<std::string> binaryNames_;
    std::vector<std::pair<std::string, Binary>> binaries_; // <binaryName, {binaryOffset, binaryLength}>
    std::vector<char> binariesBuffer_;
};
} // namespace Mki
#endif
