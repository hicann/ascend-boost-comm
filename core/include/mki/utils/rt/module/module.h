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
#ifndef MKI_UTILS_RT_MODULE_MODULE_H
#define MKI_UTILS_RT_MODULE_MODULE_H

#include "mki/utils/rt/base/types.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace Mki {
int MkiRtModuleCreate(MkiRtModuleInfo *moduleInfo, MkiRtModule *module);
int MkiRtModuleCreateFromFile(const char *moduleFilePath, MkiRtModuleType type, int version, MkiRtModule *module);
int MkiRtModuleDestory(MkiRtModule *module);
int MkiRtModuleBindFunction(MkiRtModule module, const char *funcName, void *func);
int AstRtRegisterAllFunction(MkiRtModuleInfo *moduleInfo, void **handle);
int MkiRtFunctionLaunch(const void *func, const MkiRtKernelParam *launchParam, MkiRtStream stream);
int MkiRtFunctionLaunchWithHandle(void *handle, const MkiRtKernelParam *launchParam, MkiRtStream stream,
    const RtTaskCfgInfoT *cfgInfo);
int MkiRtFunctionLaunchWithFlag(const void *func, const MkiRtKernelParam *launchParam, MkiRtStream stream,
    const RtTaskCfgInfoT *cfgInfo);
}
#ifdef __cplusplus
}
#endif
#endif
