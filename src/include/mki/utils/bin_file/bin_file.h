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
#ifndef MKI_UTILS_BINFILE_BINFILE_H
#define MKI_UTILS_BINFILE_BINFILE_H
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
