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
#include <thread>
#include <mutex>
#include <dlfcn.h>
#include <mki/utils/singleton/singleton.h>
#include <mki/utils/log/log.h>
#include "mki/utils/profiling/prof_api.h"
#include "mki/utils/profiling/profiling_funcs.h"

namespace Mki {

bool ProfilingFuncs::isProfilingLevel0Enable_ = false;
bool ProfilingFuncs::isProfilingLevel1Enable_ = false;

static ProfilingFuncs g_profiling = Mki::GetSingleton<ProfilingFuncs>();

void ProfilingFuncs::SetProfilingLevel0Status(bool status)
{
    isProfilingLevel0Enable_ = status;
}

void ProfilingFuncs::SetProfilingLevel1Status(bool status)
{
    isProfilingLevel1Enable_ = status;
}

bool ProfilingFuncs::GetProfilingLevel0Status()
{
    return isProfilingLevel0Enable_;
};

bool ProfilingFuncs::GetProfilingLevel1Status()
{
    return isProfilingLevel1Enable_;
}

ProfilingFuncs::ProfilingFuncs() noexcept
{
    MKI_LOG(INFO) << "MsprofRegisterCallback start!";
    if (MsprofRegisterCallback(0, ProfilingFuncs::MkiProfCommandHandle) != 0) {
        MKI_LOG(ERROR) << "MsprofRegisterCallback fail!";
    }
}

ProfilingFuncs::~ProfilingFuncs() {}

int32_t ProfilingFuncs::ProfReportApi(uint32_t agingFlag, const MsProfApi *api) const
{
    MKI_LOG(INFO) << "ProfReportApi start!";
    return MsprofReportApi(agingFlag, api);
}

int32_t ProfilingFuncs::ProfReportCompactInfo(uint32_t agingFlag, const void *data, uint32_t length) const
{
    MKI_LOG(INFO) << "ProfReportCompactInfo start!";
    if (isProfilingLevel1Enable_) {
        return MsprofReportCompactInfo(agingFlag, data, length);
    }
    return 0;
}

int32_t ProfilingFuncs::ProfReportAdditionalInfo(uint32_t agingFlag, const void *data, uint32_t length) const
{
    MKI_LOG(INFO) << "ProfReportAdditionalInfo start!";
    return MsprofReportAdditionalInfo(agingFlag, data, length);
}

uint64_t ProfilingFuncs::ProfSysCycleTime() const
{
    return MsprofSysCycleTime();
}

uint64_t ProfilingFuncs::ProfGetHashId(const char *hashInfo, size_t length) const
{
    return MsprofGetHashId(hashInfo, length);
}

uint64_t ProfilingFuncs::ProfGetHashId(const char *hashInfo, size_t length, void const *key)
{
    auto it = kernelNameHashCache_.find(key);
    if (it == kernelNameHashCache_.end()) {
        auto hashId = MsprofGetHashId(hashInfo, length);
        kernelNameHashCache_.insert({key, hashId});
        return hashId;
    }

    return it->second;
}

int32_t ProfilingFuncs::MkiProfCommandHandle(uint32_t type, void *data, uint32_t len)
{
    MKI_LOG(DEBUG) << "Step in MkiProfCommandHandle!";
    if (data == nullptr) {
        MKI_LOG(ERROR) << "MkiProfCommandHandle failed! data is nullptr!";
        return PROFILING_REPORT_FAILED;
    }
    if (type != PROF_CTRL_SWITCH) {
        MKI_LOG(ERROR) << "MkiProfCommandHandle failed! ProfCtrlType is not correct!";
        return PROFILING_REPORT_FAILED;
    }

    if (len < sizeof(MsprofCommandHandle)) {
        MKI_LOG(ERROR) << "MkiProfCommandHandle failed! dataSize is not correct!";
        return PROFILING_REPORT_FAILED;
    }
    MsprofCommandHandle *profilerConfig = static_cast<MsprofCommandHandle*>(data);
    const uint64_t profSwitch = profilerConfig->profSwitch;
    const uint32_t profType = profilerConfig->type;

    if (profType == PROF_COMMANDHANDLE_TYPE_START) {
        MKI_LOG(INFO) << "Open Profiling Switch";
        isProfilingLevel0Enable_ = true;
    }
    if (profType == PROF_COMMANDHANDLE_TYPE_STOP) {
        MKI_LOG(INFO) << "Close Profiling Switch";
        isProfilingLevel0Enable_ = false;
    }
    if (((profSwitch & PROF_TASK_TIME_L1) != PROF_CTRL_INVALID) ||
        ((profSwitch & PROF_TASK_TIME_L2) != PROF_CTRL_INVALID)) {
        MKI_LOG(INFO) << "Open Profiling TensorInfo Switch";
        isProfilingLevel1Enable_ = true;
    } else {
        isProfilingLevel1Enable_ = false;
    }

    return PROFILING_REPORT_SUCCESS;
}
} // namespace Mki
