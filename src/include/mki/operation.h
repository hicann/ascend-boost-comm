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
#ifndef MKI_OPERATION_H
#define MKI_OPERATION_H

#include <string>
#include <vector>
#include <unordered_map>
#include "mki/launch_param.h"
#include "mki/kernel.h"
#include "mki/utils/any/any.h"
#include "mki/utils/status/status.h"

namespace Mki {
using KernelList = std::vector<Kernel const *>;
using KernelMap = std::unordered_map<std::string, Kernel const *>;
class Operation {
public:
    Operation() = default;
    virtual ~Operation() = default;
    virtual std::string GetName() const = 0;
    virtual Status InferShape(LaunchParam &launchParam) const = 0;

    virtual int64_t GetInputNum(const Any &specificParam) const = 0;
    virtual int64_t GetOutputNum(const Any &specificParam) const = 0;
    virtual bool IsConsistent(const LaunchParam &launchParam) const = 0;

    virtual const KernelList &GetKernelList() const = 0;
    virtual Kernel *GetBestKernel(const LaunchParam &launchParam) const = 0;
    virtual Kernel *GetKernelByName(const std::string &kernelName) const = 0;
};
} // namespace Mki

#endif
