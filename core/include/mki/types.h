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
#ifndef MKI_TYPES_H
#define MKI_TYPES_H

#include <cstddef>
#include <string>

namespace Mki {
enum TensorDType : int {
    TENSOR_DTYPE_UNDEFINED = -1,
    TENSOR_DTYPE_FLOAT = 0,
    TENSOR_DTYPE_FLOAT16 = 1,
    TENSOR_DTYPE_INT8 = 2,
    TENSOR_DTYPE_INT32 = 3,
    TENSOR_DTYPE_UINT8 = 4,
    TENSOR_DTYPE_INT16 = 6,
    TENSOR_DTYPE_UINT16 = 7,
    TENSOR_DTYPE_UINT32 = 8,
    TENSOR_DTYPE_INT64 = 9,
    TENSOR_DTYPE_UINT64 = 10,
    TENSOR_DTYPE_DOUBLE = 11,
    TENSOR_DTYPE_BOOL = 12,
    TENSOR_DTYPE_STRING = 13,
    TENSOR_DTYPE_COMPLEX64 = 16,
    TENSOR_DTYPE_COMPLEX128 = 17,
    TENSOR_DTYPE_BF16 = 27
};

enum TensorFormat : int {
    TENSOR_FORMAT_UNDEFINED = -1,
    TENSOR_FORMAT_NCHW = 0,
    TENSOR_FORMAT_NHWC = 1,
    TENSOR_FORMAT_ND = 2,
    TENSOR_FORMAT_NC1HWC0 = 3,
    TENSOR_FORMAT_FRACTAL_Z = 4,
    TENSOR_FORMAT_NC1HWC0_C04 = 12,
    TENSOR_FORMAT_HWCN = 16,
    TENSOR_FORMAT_NDHWC = 27,
    TENSOR_FORMAT_FRACTAL_NZ = 29,
    TENSOR_FORMAT_NCDHW = 30,
    TENSOR_FORMAT_NDC1HWC0 = 32,
    TENSOR_FORMAT_FRACTAL_Z_3D = 33
};

enum ErrorType : int {
    NO_ERROR = 0,
    ERROR_INVALID_VALUE = 1,
    ERROR_OPERATION_NOT_EXIST = 2,
    ERROR_TACTIC_NOT_EXIST = 3,
    ERROR_KERNEL_NOT_EXIST = 4,
    ERROR_ATTR_NOT_EXIST = 5,
    ERROR_ATTR_INVALID_TYPE = 6,
    ERROR_LAUNCH_KERNEL_ERROR = 7,
    ERROR_SYNC_STREAM_ERROR = 8,
    ERROR_INFERSHAPE_ERROR = 9,
    ERROR_NOT_CONSISTANT = 10
};

size_t GetTensorElementSize(const TensorDType dtype);
int GetDTypeWithStr(const std::string &typeStr);
int GetFormatWithStr(const std::string &formatStr);
const std::string& GetStrWithDType(const int &dType);
const std::string& GetStrWithFormat(const int &format);
} // namespace Mki

#endif
