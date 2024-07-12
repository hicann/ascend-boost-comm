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
#ifndef MKI_UTILS_RT_BASE_TYPES_H
#define MKI_UTILS_RT_BASE_TYPES_H
#include <cstdint>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *MkiDevice;
typedef void *MkiRtStream;

typedef enum {
    MKIRT_SUCCESS = 0,
    MKIRT_ERROR_NOT_INITIALIZED = -1,
    MKIRT_ERROR_NOT_IMPLMENT = -2,
    MKIRT_ERROR_ASCEND_ENV_NOT_EXIST = -3,
    MKIRT_ERROR_LOAD_RUNTIME_FAIL = -4,
    MKIRT_ERROR_FUNC_NOT_EXIST = -5,
    MKIRT_ERROR_OPEN_BIN_FILE_FAIL = -6,
    MKIRT_ERROR_PARA_CHECK_FAIL = -7,
} MkiRtError;

typedef enum {
    MKIRT_MEMCOPY_HOST_TO_HOST = 0,
    MKIRT_MEMCOPY_HOST_TO_DEVICE,
    MKIRT_MEMCOPY_DEVICE_TO_HOST,
    MKIRT_MEMCOPY_DEVICE_TO_DEVICE, // device to device, 1P &P2P
    MKIRT_MEMCOPY_MANAGED,
    MKIRT_MEMCOPY_ADDR_DEVICE_TO_DEVICE,
    MKIRT_MEMCOPY_HOST_TO_DEVICE_EX, // host to device ex(only used for 8 bytes)
    MKIRT_MEMCOPY_DEVICE_TO_HOST_EX, // device to host ex
} MkiRtMemCopyType;

typedef enum {
    MKIRT_MEM_DEFAULT = 0,    // default memeory on device
    MKIRT_MEM_HBM = 0x2,      // HBM memory on device
    MKIRT_MEM_RDMA_HBM = 0x3, // RDMA-HBM memory on device
    MKIRT_MEM_DDR = 0x4,      // DDR memory on device
    MKIRT_MEM_SPM = 0x8,      // shaed physical memory on device
    MKIRT_MEM_P2P_HBM = 0x10, // HBM memory on other 4p device
    MKIRT_MEM_P2P_DDR = 0x11, // DDR memory on other device
    MKIRT_MEM_DDR_NC = 0x20,  // DDR memory of non-cache
    MKIRT_MEM_TS_4G = 0x40,
    MKIRT_MEM_TS = 0x80,
    MKIRT_MEM_HOST = 0x81, // memory on host
    MKIRT_MEM_RESERVED = 0x100,
    MKIRT_MEM_L1 = (0x1 << 16),
    MKIRT_MEM_L2 = (0x1 << 17),
} MkiRtMemType;

typedef void *MkiRtModule;

typedef enum {
    MKIRT_MODULE_OBJECT = 0, // 原始object文件
    MKIRT_MODULE_BIN = 1,    // bin头 + 连续原始object
    MKIRT_MODULE_FUSEIO_BIN  // fusionbin头 + 连续的bin
} MkiRtModuleType;

typedef struct {
    MkiRtModuleType type = MKIRT_MODULE_OBJECT;
    uint32_t version = 0;
    const void *data = nullptr;
    uint64_t dataLen = 0;
    uint32_t magic = 0x41494343U;
} MkiRtModuleInfo;

typedef struct {
    uint16_t addrOffset{0};
    uint16_t dataOffset{0};
} RtHostInputInfoT;

typedef struct {
    void *args{nullptr};
    RtHostInputInfoT *hostInputInfoPtr{nullptr};
    uint32_t argsSize{0};
    uint16_t tilingAddrOffset{0};
    uint16_t tilingDataOffset{0};
    uint16_t hostInputInfoNum{0};
    uint8_t hasTiling{0};
    uint8_t isNoNeedH2DCopy{0};
    uint8_t reserved[4] = {0};
} RtArgsExT;

typedef struct {
    uint8_t qos{0};
    uint8_t partId{0};
    uint8_t schemMode{0};
    uint8_t res[1] = {0};
} RtTaskCfgInfoT;

typedef struct {
    uint64_t tilingId = 0;
    uint32_t blockDim = 0;
    void *args = nullptr;
    uint32_t argSize = 0;
    RtArgsExT *argsEx = nullptr;
} MkiRtKernelParam;

#ifdef __cplusplus
}
#endif
#endif
