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

void LaunchParam::SetInputLens(SVector<int> &Lens) { inputLens_ = Lens; }

size_t LaunchParam::GetInputLenCount() const { return inputLens_.size(); }

int LaunchParam::GetInputLen(size_t pos) const { return inputLens_.at(pos); }

const SVector<int> &LaunchParam::GetInputLens() const { return inputLens_; }

void LaunchParam::SetOutputLens(SVector<int> &Lens) { outputLens_ = Lens; }

size_t LaunchParam::GetOutputLenCount() const { return outputLens_.size(); }

int LaunchParam::GetOutputLen(size_t pos) const { return outputLens_.at(pos); }

const SVector<int> &LaunchParam::GetOutputLens() const { return outputLens_; }

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
