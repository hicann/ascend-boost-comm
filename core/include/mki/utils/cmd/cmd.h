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
#ifndef MKI_CMD_CMD_H
#define MKI_CMD_CMD_H
#include <vector>
#include <functional>
#include <map>
#include <string>
#include "mki/utils/status/status.h"

namespace Mki {
using CmdFunc = std::function<Status(const std::vector<std::string> &args)>;

class Cmd {
public:
    Cmd();
    ~Cmd() = default;
    void Register(const std::string &cmd, CmdFunc func, const std::string &desc);
    Status Run(int argc, const char *argv[]);

private:
    Status DoHelp(const std::vector<std::string> &args) const;

private:
    struct CmdItem {
        CmdFunc func;
        std::string desc;
    };
    std::map<std::string, CmdItem> cmdMap_;
};
} // namespace Mki
#endif
