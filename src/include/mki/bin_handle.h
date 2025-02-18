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
#ifndef MKI_BIN_HANDLE_H
#define MKI_BIN_HANDLE_H
#include <string>
#include <atomic>
#include "mki/tensor.h"
#include "mki/utils/non_copyable/non_copyable.h"

namespace Mki {
using KernelHandle = void *const *;

struct BinaryBasicInfo {
    const uint8_t *binaryBuf = nullptr;
    uint32_t binaryLen = 0;
    std::string targetSoc;
};

struct KernelMetaInfo {
    std::vector<std::string> kernelList;
    std::string compileInfo;
    uint32_t version = 0;
    uint32_t tilingSize = 0;
    uint32_t coreType = 0;
    uint32_t magic = 0;
    uint32_t intercoreSync = 0;
    uint32_t cubeRatio = 0;
    uint32_t vectorRatio = 0;
    const void *codeBuf = nullptr;
    uint32_t codeBufLen = 0;
};

class BinHandle : public NonCopyable {
public:
    explicit BinHandle(const BinaryBasicInfo *binInfo);
    ~BinHandle();
    KernelHandle GetHandle() const;
    bool Init(const std::string &kernelName);
    uint32_t GetKernelTilingSize() const;
    int32_t GetKernelCoreType() const;
    uint32_t GetIntercoreSync() const;
    uint32_t GetCubeRatio() const;
    uint32_t GetVectorRatio() const;
    const char *GetKernelCompileInfo() const;

private:
    bool CheckBinaryValid() const;
    bool CheckKernelInfo(const std::string &kernelName) const;
    bool RegisterBin(const std::string &kernelName);

private:
    KernelMetaInfo metaInfo_;
    void *handle_ = nullptr;
    void *moduleHandle_ = nullptr;
    BinaryBasicInfo const *const basicInfo_ = nullptr;
    std::atomic_bool codeLoadSuccess_ = false;
};
} // namespace Mki

#endif
