/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
