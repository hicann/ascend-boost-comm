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
#include "mki/utils/cmd/cmd.h"
#include <iostream>
#include "mki/utils/assert/assert.h"

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
    UNUSED_VALUE(args);
    for (const auto &it : cmdMap_) {
        if (it.first == CMD_HELP) {
            continue;
        }
        std::cout << it.second.desc << std::endl;
    }
    return Status::OkStatus();
}
} // namespace Mki
