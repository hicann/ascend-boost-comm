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
