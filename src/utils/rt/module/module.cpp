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
#include "mki/utils/rt/module/module.h"
#include "mki/utils/rt/backend/backend_factory.h"

namespace Mki {
int MkiRtModuleCreate(MkiRtModuleInfo *moduleInfo, MkiRtModule *module)
{
    return BackendFactory::GetBackend()->ModuleCreate(moduleInfo, module);
}

int MkiRtModuleCreateFromFile(const char *moduleFilePath, MkiRtModuleType type, int version, MkiRtModule *module)
{
    return BackendFactory::GetBackend()->ModuleCreateFromFile(moduleFilePath, type, version, module);
}

int MkiRtModuleDestory(MkiRtModule *module) { return BackendFactory::GetBackend()->ModuleDestory(module); }

int MkiRtModuleBindFunction(MkiRtModule module, const char *funcName, void *func)
{
    return BackendFactory::GetBackend()->ModuleBindFunction(module, funcName, func);
}

int MkiRtRegisterAllFunction(MkiRtModuleInfo *moduleInfo, void **handle)
{
    return BackendFactory::GetBackend()->RegisterAllFunction(moduleInfo, handle);
}

int MkiRtFunctionLaunch(const void *func, const MkiRtKernelParam *launchParam, MkiRtStream stream)
{
    return BackendFactory::GetBackend()->FunctionLaunch(func, launchParam, stream);
}

int MkiRtFunctionLaunchWithHandle(void *handle, const MkiRtKernelParam *launchParam, MkiRtStream stream,
    const RtTaskCfgInfoT *cfgInfo)
{
    return BackendFactory::GetBackend()->FunctionLaunchWithHandle(handle, launchParam, stream, cfgInfo);
}

int MkiRtFunctionLaunchWithFlag(const void *func, const MkiRtKernelParam *launchParam, MkiRtStream stream,
    const RtTaskCfgInfoT *cfgInfo)
{
    return BackendFactory::GetBackend()->FunctionLaunchWithFlag(func, launchParam, stream, cfgInfo);
}
}
