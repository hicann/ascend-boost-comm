/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/launch_param.h"
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "mki/utils/stringify/stringify.h"

namespace Mki {
using ToStringFunc = std::function<std::string(const Any &)>;

LaunchParam::LaunchParam(const LaunchParam &other) { *this = other; }

LaunchParam &LaunchParam::operator=(const LaunchParam &other)
{
    specificParam_ = other.specificParam_;
    inTensors_.resize(other.inTensors_.size());
    for (size_t i = 0; i < other.inTensors_.size(); ++i) {
        inTensors_[i] = other.inTensors_[i];
    }

    outTensors_.resize(other.outTensors_.size());
    for (size_t i = 0; i < other.outTensors_.size(); ++i) {
        outTensors_[i] = other.outTensors_[i];
    }

    return *this;
}

LaunchParam::~LaunchParam() {}

void LaunchParam::Reset()
{
    specificParam_.Reset();
    inTensors_.clear();
    outTensors_.clear();
}

void LaunchParam::SetParam(const Any &srcParam) { specificParam_ = srcParam; }

void LaunchParam::AddInTensor(const Tensor &tensor) { inTensors_.push_back(tensor); }

size_t LaunchParam::GetInTensorCount() const { return inTensors_.size(); }

Tensor &LaunchParam::GetInTensor(size_t pos) { return inTensors_.at(pos); }

const Tensor &LaunchParam::GetInTensor(size_t pos) const { return inTensors_.at(pos); }

const SVector<Tensor> &LaunchParam::GetInTensors() const { return inTensors_; }

SVector<Tensor> &LaunchParam::GetInTensors() { return inTensors_; }

void LaunchParam::AddOutTensor(const Tensor &tensor) { outTensors_.push_back(tensor); }

size_t LaunchParam::GetOutTensorCount() const { return outTensors_.size(); }

Tensor &LaunchParam::GetOutTensor(size_t pos) { return outTensors_.at(pos); }

const Tensor &LaunchParam::GetOutTensor(size_t pos) const { return outTensors_.at(pos); }

const SVector<Tensor> &LaunchParam::GetOutTensors() const { return outTensors_; }

SVector<Tensor> &LaunchParam::GetOutTensors() { return outTensors_; }

std::string LaunchParam::ToString() const
{
    std::stringstream ss;
    ss << "specificParam: " << Stringify::ToString(specificParam_) << std::endl;
    for (size_t i = 0; i < inTensors_.size(); ++i) {
        ss << "intensors[" << i << "]: " << inTensors_.at(i).ToString() << std::endl;
    }
    for (size_t i = 0; i < outTensors_.size(); ++i) {
        ss << "outtensors[" << i << "]: " << outTensors_.at(i).ToString() << std::endl;
    }
    ss << std::endl;
    return ss.str();
}
} // namespace Mki
