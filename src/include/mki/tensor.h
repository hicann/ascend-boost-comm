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
#ifndef MKI_TENSOR_H
#define MKI_TENSOR_H
#include <string>
#include "mki/types.h"
#include "mki/utils/SVector/SVector.h"

namespace Mki {
struct TensorDesc {
    TensorDType dtype = TENSOR_DTYPE_UNDEFINED;
    TensorFormat format = TENSOR_FORMAT_UNDEFINED;
    Mki::SVector<int64_t> dims;
    int64_t Numel() const;
    void View(const Mki::SVector<int64_t> &newDims);
    std::string ToString() const;
};

struct Tensor {
    TensorDesc desc;
    void *data = nullptr;
    size_t dataSize = 0;
    void *hostData = nullptr;
    int64_t Numel() const;
    void View(const Mki::SVector<int64_t> &newDims);
    std::string ToString() const;
};
} // namespace Mki

#endif
