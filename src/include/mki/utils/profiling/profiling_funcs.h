/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendTransformerBoost is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef PROFILING_FUNCS_H
#define PROFILING_FUNCS_H

#include <mutex>
#include <unordered_map>
#include "prof_api.h"

namespace Mki {
class ProfilingFuncs {
public:
    ProfilingFuncs() noexcept;
    ~ProfilingFuncs();
    int32_t ProfReportApi(uint32_t agingFlag, const MsProfApi *api) const;
    int32_t ProfReportCompactInfo(uint32_t agingFlag, const void *data, uint32_t length) const;
    int32_t ProfReportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length) const;
    uint64_t ProfSysCycleTime() const;
    uint64_t ProfGetHashId(const char *hashInfo, size_t length) const;
    uint64_t ProfGetHashId(const char *hashInfo, size_t length, void const *key);
    void SetProfilingLevel0Status(bool status) const;
    void SetProfilingLevel1Status(bool status) const;
    bool GetProfilingLevel0Status() const;
    bool GetProfilingLevel1Status() const;
    static int32_t MkiProfCommandHandle(uint32_t type, void *data, uint32_t len);

private:
    std::unordered_map<void const *, uint64_t> kernelNameHashCache_;
    static bool isProfilingLevel0Enable_;
    static bool isProfilingLevel1Enable_;
};
} // namespace Mki
#endif
