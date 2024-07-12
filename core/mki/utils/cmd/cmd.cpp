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
#include "mki/utils/cmd/cmd.h"
#include <iostream>

namespace Mki {
constexpr int ERROR_INVALID_ARG = 1;
constexpr int ARG_LEN_MIN = 2;
const std::string CMD_HELP = "help";

Cmd::Cmd() { cmdMap_[CMD_HELP] = {std::bind(&Cmd::DoHelp, this, std::placeholders::_1), ""}; }

void Cmd::Register(const std::string &cmd, CmdFunc func, const std::string &desc) { cmdMap_[cmd] = {func, desc}; }

Status Cmd::Run(int argc, const char *argv[])
{
    if (argc < ARG_LEN_MIN) {
        return Status::FailStatus(ERROR_INVALID_ARG, "invalid args");
    }
    if (argv == nullptr) {
        return Status::FailStatus(ERROR_INVALID_ARG, "argv is nullptr");
    }
    std::string cmd = argv[1];
    const auto it = cmdMap_.find(cmd);
    if (it == cmdMap_.cend()) {
        return Status::FailStatus(ERROR_INVALID_ARG, "not support cmd:" + cmd);
    }

    std::vector<std::string> args;
    for (int i = 2; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    return it->second.func(args);
}

Status Cmd::DoHelp(const std::vector<std::string> &args) const
{
    (void)args;
    for (const auto &it : cmdMap_) {
        if (it.first == CMD_HELP) {
            continue;
        }
        std::cout << it.second.desc << std::endl;
    }
    return Status::OkStatus();
}
} // namespace Mki
