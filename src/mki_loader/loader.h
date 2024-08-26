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
#ifndef MKI_LOADER_LOADER_H
#define MKI_LOADER_LOADER_H
#include <unordered_map>
#include <memory>
#include <atomic>
#include "mki/utils/non_copyable/non_copyable.h"
#include "mki/operation.h"
#include "mki/kernel.h"
#include "mki/bin_handle.h"

namespace OpSpace {
class Loader : public Mki::NonCopyable {
public:
    Loader();
    ~Loader();
    void GetAllOperations(std::unordered_map<std::string, Mki::Operation *> &ops) const;
    void GetOpKernels(const std::string &opName, Mki::KernelMap &kernels) const;
    bool IsValid() const;

private:
    void Load();
    bool LoadKernelBinarys();
    bool CreateOperations();
    bool CreateKernels();
    bool OpBaseAddKernels() const;

private:
    std::atomic_bool loadSuccess_{false};
    std::unordered_map<std::string, Mki::Operation *> opMap_;
    std::unordered_map<std::string, Mki::KernelMap> opKernelMap_;
    std::unordered_map<std::string, Mki::BinHandle> binHandles_;
};
} // namespace OpSpace
#endif
