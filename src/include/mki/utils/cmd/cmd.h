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
#ifndef MKI_UTILS_CMD_CMD_H
#define MKI_UTILS_CMD_CMD_H
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
