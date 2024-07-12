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
#ifndef MKI_TENSORDESC_H
#define MKI_TENSORDESC_H
#include "mki/types.h"
#include "mki/utils/svector/svector.h"

namespace Mki {
struct TensorDesc {
    TensorDType dtype = TENSOR_DTYPE_UNDEFINED;
    TensorFormat format = TENSOR_FORMAT_UNDEFINED;
    Mki::SVector<int64_t> dims;
    int64_t Numel() const;
    void View(const Mki::SVector<int64_t> &newDims);
    std::string ToString() const;
};
} // namespace Mki

#endif
