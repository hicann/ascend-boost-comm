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
#include <functional>
#include <map>
#include "mki/utils/log/log.h"
#include "mki/launch_param.h"

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

std::string LaunchParam::ParamToJsonString() const
{
    MKI_LOG(WARN) << "LaunchParam can not print param " << specificParam_.Type().name();
    return "";
}

std::string LaunchParam::ToString() const
{
    std::stringstream ss;
    ss << "specificParam: " << ParamToJsonString() << std::endl;
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
