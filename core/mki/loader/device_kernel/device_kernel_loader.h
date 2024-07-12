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
#ifndef CORE_LOADER_DEVICEKERNELLOADER_H
#define CORE_LOADER_DEVICEKERNELLOADER_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include "mki/base/op_register.h"

namespace Mki {
struct KernelMetaInfo {
    std::vector<std::string> kernelList;
    std::string compileInfo;
    uint32_t version;
    uint32_t tilingSize = 0;
    uint32_t coreType;
    uint32_t magic = 0x41494343U;
    void *codeBuf = nullptr;
    uint32_t codeBufLen = 0;
};

class KernelBinaryLoader {
public:
    KernelBinaryLoader();
    virtual ~KernelBinaryLoader();
    size_t GetBinaryKernelNum(const std::string &opName);
    int GetBinaryKernelList(const std::string &opName, std::vector<std::string> &kernelList);
    int GetBinaryDeviceCode(const std::string &opName, void *&deviceCode, uint32_t &bufLen);
    const char *GetKernelCompileInfo(const std::string &opName);
    uint32_t GetKernelTilingSize(const std::string &opName);
    int32_t GetKernelCoreType(const std::string &opName);
    uint32_t GetKernelMagic(const std::string &opName);
    bool HasKernelMetaInfo(const std::string &opName);

private:
    void Init();
    void CheckKernelInfo();
    bool CheckBinaryValid(BinaryBasicInfo &binaryBasicInfo);
    std::atomic_bool codeLoadSuccess_{false};

private:
    std::unordered_map<std::string, KernelMetaInfo> opInfoMap_;
};
} // namespace Mki
#endif
