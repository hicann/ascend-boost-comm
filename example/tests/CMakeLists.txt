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
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/autogen
    $ENV{PYTHON_INCLUDE_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/include
    $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
    $ENV{PYTORCH_NPU_INSTALL_PATH}/include
)

link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
)

add_subdirectory(autogen)

install(DIRECTORY ${MKI_PACKAGE_DIR}/tests DESTINATION .
        PATTERN "libmki_test_autogen.so" EXCLUDE)