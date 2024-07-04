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
#ifndef MKI_OPS_H
#define MKI_OPS_H
#include <vector>
#include <string>
#include <memory>
#include "mki/kernel.h"
#include "mki/operation.h"
#include "mki/tensor.h"
#include "mki/run_info.h"
#include "mki/op_schedule.h"

namespace Mki {

class Ops {
public:
    /**
     * @brief Return the singleton object
     *
     * @return Ops&
     */
    static Ops &Instance();
    /**
     * @brief Get the All Operations object
     *
     * @return std::vector<Operation *> &
     */
    std::vector<Operation *> GetAllOperations() const;
    /**
     * @brief Get the Operation By Name object
     *
     * @param[const std::string&] opName
     * @return Operation*
     */
    Operation *GetOperationByName(const std::string &opName) const;
    /**
     * @brief Get the Kernel Instance By Name
     *
     * @param[const std::string&] opName
     * @return Kernel*
     */
    Kernel *GetKernelInstance(const std::string &opName) const;

private:
    Ops();
    ~Ops();

private:
    std::unique_ptr<OpSchedule> opSchedule_;
};
} // namespace Mki

#endif
