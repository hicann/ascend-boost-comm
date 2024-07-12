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

int AstRtRegisterAllFunction(MkiRtModuleInfo *moduleInfo, void **handle)
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
