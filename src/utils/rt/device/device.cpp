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
#include "mki/utils/rt/device/device.h"
#include "mki/utils/rt/backend/backend_factory.h"

namespace Mki {
int MkiRtDeviceGetCount(int32_t *devCount) { return BackendFactory::GetBackend()->DeviceGetCount(devCount); }

int MkiRtDeviceGetIds(int32_t *devIds, int32_t devIdNum)
{
    return BackendFactory::GetBackend()->DeviceGetIds(devIds, devIdNum);
}

int MkiRtDeviceGetCurrent(int32_t *devId) { return BackendFactory::GetBackend()->DeviceGetCurrent(devId); }

int MkiRtDeviceSetCurrent(int32_t devId) { return BackendFactory::GetBackend()->DeviceSetCurrent(devId); }

int MkiRtDeviceResetCurrent(int32_t devId) { return BackendFactory::GetBackend()->DeviceResetCurrent(devId); }

int MkiRtDeviceSetSocVersion(const char *version)
{
    return BackendFactory::GetBackend()->DeviceSetSocVersion(version);
}

int MkiRtDeviceGetSocVersion(char *version, uint32_t maxLen)
{
    return BackendFactory::GetBackend()->DeviceGetSocVersion(version, maxLen);
}

int MkiRtDeviceGetBareTgid(uint32_t *pid)
{
    return BackendFactory::GetBackend()->DeviceGetBareTgid(pid);
}

int MkiRtDeviceGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t *val)
{
    return BackendFactory::GetBackend()->DeviceGetPairDevicesInfo(devId, otherDevId, infoType, val);
}
}
