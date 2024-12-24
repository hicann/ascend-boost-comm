/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mki/utils/rt/device/device.h"
#include "mki/utils/rt/backend/backend_factory.h"
#include <acl/acl.h>
#include "mki/utils/log/log.h"
#include <cstring>



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

int MkiRtDeviceGetSocVersion(std::string &version, uint32_t maxLen)
{
    const char* version2 = aclrtGetSocName();

    version = version2;

    MKI_LOG(INFO) << "DeviceVersion1: " << version; // 打印 version
    MKI_LOG(INFO) << "DeviceVersion2: " << version2; // 打印 version2

    if (strlen(version2) == 0) {
        return 1;
    }

    return 0;
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
