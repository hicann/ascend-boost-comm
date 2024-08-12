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
#ifndef MKI_UTILS_RT_DEVICE_DEVICE_H
#define MKI_UTILS_RT_DEVICE_DEVICE_H
#include "mki/utils/rt/base/types.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace Mki {
int MkiRtDeviceGetCount(int32_t *devCount);
int MkiRtDeviceGetIds(int32_t *devIds, int32_t devIdNum);
int MkiRtDeviceGetCurrent(int32_t *devId);
int MkiRtDeviceSetCurrent(int32_t devId);
int MkiRtDeviceResetCurrent(int32_t devId);
int MkiRtDeviceSetSocVersion(const char *version);
int MkiRtDeviceGetSocVersion(char *version, uint32_t maxLen);
int MkiRtDeviceGetBareTgid(uint32_t *pid);
int MkiRtDeviceGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val);
}
#ifdef __cplusplus
}
#endif
#endif
