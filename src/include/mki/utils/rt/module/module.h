/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * AscendOpCommonLib is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
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
