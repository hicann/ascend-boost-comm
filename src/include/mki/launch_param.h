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
#ifndef MKI_LAUNCH_PARAM_H
#define MKI_LAUNCH_PARAM_H

#include <string>
#include "mki/types.h"
#include "mki/tensor.h"
#include "mki/utils/any/any.h"

namespace Mki {
class LaunchParam {
public:
    LaunchParam() = default;
    LaunchParam(const LaunchParam &other);
    LaunchParam &operator=(const LaunchParam &other);
    ~LaunchParam();

    void Reset();

    void SetParam(const Any &srcParam);

    template <typename T> const T &GetParam() const { return AnyCast<T>(specificParam_); }
    template <typename T> T &GetParam() { return AnyCast<T>(specificParam_); }
    const Any &GetParam() const { return specificParam_; }
    Any &GetParam() { return specificParam_; }

    void AddInTensor(const Tensor &tensor);
    size_t GetInTensorCount() const;
    const Tensor &GetInTensor(size_t pos) const;
    Tensor &GetInTensor(size_t pos);
    const SVector<Tensor> &GetInTensors() const;
    SVector<Tensor> &GetInTensors();

    void AddOutTensor(const Tensor &tensor);
    size_t GetOutTensorCount() const;
    const Tensor &GetOutTensor(size_t pos) const;
    Tensor &GetOutTensor(size_t pos);
    const SVector<Tensor> &GetOutTensors() const;
    SVector<Tensor> &GetOutTensors();

    std::string ToString() const;

private:
    Any specificParam_;
    SVector<Tensor> inTensors_;
    SVector<Tensor> outTensors_;
};
} // namespace Mki

#endif
