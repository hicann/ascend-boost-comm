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
#ifndef MKI_UTILS_MEMSET_LAUNCHER_H
#define MKI_UTILS_MEMSET_LAUNCHER_H

#include <cstdint>
#include "mki/kernel_info.h"
#include "mki/utils/status/status.h"

namespace Mki {
Status ClearTensors(void **args, uint64_t argsNum, MiniVector<KernelInfo::MemsetInfo> &memsetInfo, void *stream);
} // namespace Mki

#endif // MKI_UTILS_MEMSET_LAUNCHER_H
