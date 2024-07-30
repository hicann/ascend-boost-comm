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

#ifndef UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H
#define UTILS_PLATFORM_PLATFORM_INFOS_IMPL_H

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "platform/platform_infos_def.h"

namespace fe {
class PlatFormInfosImpl {
using PlatInfoMapType = std::map<std::string, std::vector<std::string>>;

public:
    PlatInfoMapType GetAICoreIntrinsicDtype();
    PlatInfoMapType GetVectorCoreIntrinsicDtype();
    PlatInfoMapType GetFixPipeDtypeMap();

    void SetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);
    bool GetPlatformRes(const std::string &label, const std::string &key, std::string &value);
    bool GetPlatformRes(const std::string &label, std::map<std::string, std::string> &res);

    void SetFixPipeDtypeMap(const PlatInfoMapType &fixpipeDtypeMap);
    void SetAICoreIntrinsicDtype(PlatInfoMapType &intrinsicDtypes);
    void SetVectorCoreIntrinsicDtype(PlatInfoMapType &intrinsicDtypes);

private:
    PlatInfoMapType aiCoreIntrinsicDtypeMap_;
    PlatInfoMapType vectorCoreIntrinsicDtypeMap_;
    std::map<std::string, std::map<std::string, std::string>> platformResMap_;
    PlatInfoMapType fixpipeDtypeMap_;
};
} // namespace fe

#endif
