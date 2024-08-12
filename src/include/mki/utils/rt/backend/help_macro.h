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
#ifndef MKI_UTILS_RT_BACKEND_HELP_MACRO_H
#define MKI_UTILS_RT_BACKEND_HELP_MACRO_H

#define CHECK_FUNC_EIXST_RETURN(func)                                                                              \
    if ((func) == nullptr) {                                                                                       \
        return MKIRT_ERROR_FUNC_NOT_EXIST;                                                                         \
    }

#define CHECK_STATUS_RETURN(fun)                                                                                   \
    do {                                                                                                           \
        int ret = (fun);                                                                                           \
        if (ret == 0) {                                                                                            \
            return MKIRT_SUCCESS;                                                                                  \
        } else {                                                                                                   \
            return ret;                                                                                            \
        }                                                                                                          \
    } while (0)

#define CHECK_STATUS_WITH_DESC_RETURN(ret, funcName)                                                               \
    if ((ret) == 0) {                                                                                              \
        MKI_LOG(DEBUG) << (funcName) << " success";                                                                \
        return MKIRT_SUCCESS;                                                                                      \
    } else {                                                                                                       \
        MKI_LOG(ERROR) << (funcName) << " fail, error:" << (ret);                                                  \
        return ret;                                                                                                \
    }

#define CHECK_INITED_RETURN(ret)                                                                                   \
    if ((ret) != MKIRT_SUCCESS) {                                                                                  \
        return ret;                                                                                                \
    }

#define CHECK_FUN_PARA_RETURN(para)                                                                                \
    if ((para) == nullptr) {                                                                                       \
        return MKIRT_ERROR_PARA_CHECK_FAIL;                                                                        \
    }

#endif
