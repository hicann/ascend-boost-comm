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
#ifndef MKI_UTILS_RT_BACKEND_HELPMACRO_H
#define MKI_UTILS_RT_BACKEND_HELPMACRO_H

#define CHECK_FUNC_EIXST_RETURN(func)                                                                                  \
do {                                                                                                                   \
    if ((func) == nullptr) {                                                                                           \
        return MKIRT_ERROR_FUNC_NOT_EXIST;                                                                             \
    }                                                                                                                  \
} while (0)

#define CHECK_STATUS_RETURN(fun)                                                                                       \
do {                                                                                                                   \
    int ret = (fun);                                                                                                   \
    if (ret == 0) {                                                                                                    \
        return MKIRT_SUCCESS;                                                                                          \
    } else {                                                                                                           \
        return ret;                                                                                                    \
    }                                                                                                                  \
} while (0)

#define CHECK_STATUS_WITH_DESC_RETURN(ret, funcName)                                                                   \
do {                                                                                                                   \
    if ((ret) == 0) {                                                                                                  \
        MKI_LOG(DEBUG) << (funcName) << " success";                                                                    \
        return MKIRT_SUCCESS;                                                                                          \
    } else {                                                                                                           \
        MKI_LOG(ERROR) << (funcName) << " fail, error:" << (ret);                                                      \
        return ret;                                                                                                    \
    }                                                                                                                  \
} while (0)

#define CHECK_INITED_RETURN(ret)                                                                                       \
do {                                                                                                                   \
    if ((ret) != MKIRT_SUCCESS) {                                                                                      \
        return ret;                                                                                                    \
    }                                                                                                                  \
} while (0)

#define CHECK_FUN_PARA_RETURN(para)                                                                                    \
do {                                                                                                                   \
    if ((para) == nullptr) {                                                                                           \
        return MKIRT_ERROR_PARA_CHECK_FAIL;                                                                            \
    }                                                                                                                  \
} while (0)
#endif
