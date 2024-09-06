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
#ifndef MKI_UTILS_SHARE_MEMORY_SHARE_MEMORY_H
#define MKI_UTILS_SHARE_MEMORY_SHARE_MEMORY_H

#include <semaphore.h>
#include <iostream>

namespace Mki {
class ShareMemory {
public:
    ShareMemory(const std::string &name, uint32_t size);
    ~ShareMemory();
    ShareMemory(const ShareMemory &other) = delete;
    ShareMemory &operator=(const ShareMemory &other) = delete;
    void *GetShm();
    void SemLock() const;
    void SemUnLock() const;

private:
    void *CreateShareMemory(const std::string &name, uint32_t size);
    void CleanUpShm();
    void CleanUpSem();

private:
    std::string fullName_;
    sem_t *sem_ = nullptr;
    uint8_t *shareMemory_ = nullptr;
    uint32_t memSize_ = 0;
    int shmid_ = -1;
};
}

#endif
