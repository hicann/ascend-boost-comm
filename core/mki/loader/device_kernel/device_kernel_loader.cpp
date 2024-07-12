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
#include "mki/loader/device_kernel/device_kernel_loader.h"
#include <cmath>
#include <cstdlib>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
#include "mki/utils/rt/base/types.h"
#include "mki/utils/rt/device/device.h"
#include "mki/utils/strings/match.h"
#include "mki/utils/strings/str_checker.h"
namespace {
constexpr uint32_t BYTE_SIZE = 8;
constexpr uint32_t HEADER_LENGTH = 128;
constexpr uint32_t UINT32_TYPE_LENGTH = sizeof(uint32_t);
const std::unordered_map<std::string, std::string> DEVICE_VERSION_MAP {
    // 310p
    {"Ascend310P", "ascend310p"}, {"Ascend310P3", "ascend310p"},
    // 910
    {"Ascend910A", "ascend910"}, {"Ascend910", "ascend910"},
    // 910b
    {"Ascend910B", "ascend910b"}, {"Ascend910B1", "ascend910b"}, {"Ascend910B2", "ascend910b"},
    {"Ascend910B2C", "ascend910b"}, {"Ascend910B3", "ascend910b"}, {"Ascend910B4", "ascend910b"},
    // 910c
    {"Ascend910C", "ascend910b"}, {"Ascend910C1", "ascend910b"}, {"Ascend910C2", "ascend910b"},
    {"Ascend910C3", "ascend910b"}, {"Ascend910C4", "ascend910b"},
    // 310b
    {"Ascend310B1", "ascend310b"}, {"Ascend310B2", "ascend310b"}, {"Ascend310B3", "ascend310b"},
    {"Ascend310B4", "ascend310b"}
};
}

namespace Mki {
const char *DEVICE_DIR_NAME = "device";
const char *BIN_DIR_NAME = "bin";
const char *OBJECTS_PATH = "obj";
}

namespace Mki {
struct KernelHeaderInfo {
    uint32_t version;
    uint32_t magic;
    uint32_t tilingSize;
    uint32_t coreType;
    uint32_t kernelNum;
    uint32_t kernelNameOffset;
    uint32_t compileInfoOffset;
    uint32_t kernelBinOffset;
};

void KernelBinaryLoader::CheckKernelInfo()
{
    MKI_LOG(DEBUG) << "opCount: " << opInfoMap_.size();
    for (auto &item : opInfoMap_) {
        for (std::string kernelName : item.second.kernelList) {
            MKI_LOG(DEBUG) << item.first << ".kernelName: " << kernelName;
        }
        MKI_LOG(DEBUG) << item.first << ".devicecodeBufLen: " << item.second.codeBufLen;
    }
}

bool KernelBinaryLoader::CheckBinaryValid(BinaryBasicInfo &binaryBasicInfo)
{
    MKI_CHECK(binaryBasicInfo.binaryBuf != nullptr, "binary info is empty!", return false);
    MKI_CHECK(binaryBasicInfo.binaryLen > HEADER_LENGTH, "binary length is less than header length!", return false);
    return true;
}

void KernelBinaryLoader::Init()
{
    codeLoadSuccess_ = false;

    auto &binaryMap = KernelBinaryRegister::GetKernelBinaryMap();

    const uint32_t deviceVersionLen = 20;
    char deviceVersion[deviceVersionLen];
    MKI_CHECK(MkiRtDeviceGetSocVersion(deviceVersion, deviceVersionLen) == MKIRT_SUCCESS,
        "Get device version failed!", return);
    const auto item = DEVICE_VERSION_MAP.find(deviceVersion);
    MKI_CHECK(item != DEVICE_VERSION_MAP.end(), "Unrecognized device version!", return);
    std::string deviceKernelVersion = item->second;

    for (auto &item : binaryMap) {
        auto &binaryList = item.second;
        size_t i;
        for (i = 0; i < binaryList.size(); ++i) {
            if (binaryList[i].targetSoc == deviceKernelVersion) {
                break;
            }
        }
        if (i == binaryList.size()) {
            continue;
        }
        BinaryBasicInfo &binaryBasicInfo = binaryList[i];
        if (!CheckBinaryValid(binaryBasicInfo)) {
            return;
        }
        const KernelHeaderInfo &header = *reinterpret_cast<const KernelHeaderInfo *>(binaryBasicInfo.binaryBuf);
        KernelMetaInfo &meta = opInfoMap_[item.first];
        const uint8_t *data = binaryBasicInfo.binaryBuf + HEADER_LENGTH;
        const uint8_t *maxAddr = binaryBasicInfo.binaryBuf + binaryBasicInfo.binaryLen;
        meta.version = header.version;
        meta.magic = header.magic;
        meta.tilingSize = header.tilingSize;
        meta.coreType = header.coreType;
        uint32_t kernelNum = header.kernelNum;
        const uint8_t *kernelNameStart = data + header.kernelNameOffset;
        for (size_t count = 0; count < kernelNum; ++count) {
            MKI_CHECK(kernelNameStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return);
            uint32_t kernelNameSize = *reinterpret_cast<const uint32_t *>(kernelNameStart);
            kernelNameStart += UINT32_TYPE_LENGTH;
            MKI_CHECK(kernelNameStart + kernelNameSize <= maxAddr, "length error", return);
            std::string str = (const char *)kernelNameStart;
            MKI_CHECK(str.length() < kernelNameSize, "length error", return);
            meta.kernelList.push_back(str);
            kernelNameStart += kernelNameSize;
        }
        const uint8_t *compileInfoStart = data + header.compileInfoOffset;
        MKI_CHECK(compileInfoStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return);
        uint32_t compileInfoSize = *reinterpret_cast<const uint32_t *>(compileInfoStart);
        compileInfoStart += UINT32_TYPE_LENGTH;
        meta.compileInfo = (const char *)compileInfoStart;
        MKI_CHECK(meta.compileInfo.length() < compileInfoSize, "length error", return);

        const uint8_t *kernelBinStart = data + header.kernelBinOffset;
        MKI_CHECK(kernelBinStart + UINT32_TYPE_LENGTH <= maxAddr, "length error", return);
        uint32_t kernelBinSize = *reinterpret_cast<const uint32_t *>(kernelBinStart);
        MKI_CHECK(kernelBinStart + UINT32_TYPE_LENGTH + kernelBinSize == maxAddr, "length error", return);
        meta.codeBuf = const_cast<uint8_t *>(kernelBinStart + UINT32_TYPE_LENGTH);
        meta.codeBufLen = kernelBinSize;
    }
    CheckKernelInfo();
    codeLoadSuccess_ = true;
}

KernelBinaryLoader::KernelBinaryLoader() { Init(); }

KernelBinaryLoader::~KernelBinaryLoader() {}

size_t KernelBinaryLoader::GetBinaryKernelNum(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return 0;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        return item->second.kernelList.size();
    }
    return 0;
}

int KernelBinaryLoader::GetBinaryKernelList(const std::string &opName, std::vector<std::string> &kernelList)
{
    if (!codeLoadSuccess_) {
        return MKIRT_ERROR_NOT_INITIALIZED;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        kernelList = item->second.kernelList;
        return MKIRT_SUCCESS;
    }
    return MKIRT_ERROR_FUNC_NOT_EXIST;
}

int KernelBinaryLoader::GetBinaryDeviceCode(const std::string &opName, void *&deviceCode, uint32_t &bufLen)
{
    if (!codeLoadSuccess_) {
        return MKIRT_ERROR_NOT_INITIALIZED;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        bufLen = item->second.codeBufLen;
        deviceCode = item->second.codeBuf;
        return MKIRT_SUCCESS;
    }
    return MKIRT_ERROR_FUNC_NOT_EXIST;
}

const char *KernelBinaryLoader::GetKernelCompileInfo(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return nullptr;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        std::string &compileInfoStrTmp = item->second.compileInfo;
        return compileInfoStrTmp.c_str();
    }
    return nullptr;
}

uint32_t KernelBinaryLoader::GetKernelTilingSize(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return 0;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        uint32_t tilingSize = item->second.tilingSize;
        return ceil(static_cast<double>(tilingSize) / BYTE_SIZE) * BYTE_SIZE;
    }
    return 0;
}

int32_t KernelBinaryLoader::GetKernelCoreType(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return -1;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        return item->second.coreType;
    }
    return -1;
}

uint32_t KernelBinaryLoader::GetKernelMagic(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return 0;
    }
    auto item = opInfoMap_.find(opName);
    if (item != opInfoMap_.end()) {
        return item->second.magic;
    }
    return 0;
}

bool KernelBinaryLoader::HasKernelMetaInfo(const std::string &opName)
{
    if (!codeLoadSuccess_) {
        return false;
    }
    auto item = opInfoMap_.find(opName);
    return item != opInfoMap_.end();
}
} // namespace Mki
