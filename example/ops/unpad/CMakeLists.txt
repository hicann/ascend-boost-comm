#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# MindKernelInfra is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
set(unpad_srcs
    ${CMAKE_CURRENT_LIST_DIR}/unpad_operation.cpp
    ${CMAKE_CURRENT_LIST_DIR}/unpad_kernel.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tiling/unpad_tiling.cpp
)

add_operation(UnpadOperation "${unpad_srcs}")


add_kernel(unpad ascend910b vector
    op_kernel/kernel_unpad.cpp
    UnpadKernel)

add_kernel(unpad ascend310p vector
    op_kernel/kernel_unpad.cpp
    UnpadKernel)
