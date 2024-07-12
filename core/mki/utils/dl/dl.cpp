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
#include "mki/utils/dl/dl.h"
#include <atomic>
#include <unistd.h>
#include <dlfcn.h>
#include "mki/utils/assert/assert.h"
#include "mki/utils/log/log.h"
#include "mki/utils/filesystem/filesystem.h"
namespace Mki {
std::atomic_int g_memfdNextId(0);

Dl::Dl(const std::string &dlFilePath) : memFd_(0), dlHandle_(nullptr)
{
    dlSuccess_ = false;
    std::string realPath = FileSystem::PathCheckAndRegular(dlFilePath);
    MKI_CHECK(!realPath.empty(), "file path is invalid", return);
    MKI_CHECK((FileSystem::FileSize(realPath) > 0 && FileSystem::FileSize(realPath) <= MAX_FILE_SIZE / 10),
        "File size is invalid", return);   // MAX_FILE_SIZE / 10 = 1GB
    dlHandle_ = dlopen(realPath.c_str(), RTLD_LAZY);
    MKI_LOG_IF(dlHandle_ == nullptr, ERROR) << "dlopen " << dlFilePath << " fail, error:" << dlerror();
    dlSuccess_ = true;
}

Dl::~Dl()
{
    if (dlHandle_) {
        dlclose(dlHandle_);
        dlHandle_ = nullptr;
    }

    if (memFd_ > 0) {
        close(memFd_);
        memFd_ = 0;
    }
}

bool Dl::IsValid() const
{
    return dlSuccess_;
}

void *Dl::GetSymbol(const std::string &symbolName) const
{
    if (dlHandle_ == nullptr) {
        return nullptr;
    }

    return dlsym(dlHandle_, symbolName.c_str());
}

std::string Dl::GetNewMemFdFileName() const
{
    std::string fileName;
    int id = g_memfdNextId++;
    fileName = "dl_" + std::to_string(id);
    return fileName;
}
} // namespace Mki
