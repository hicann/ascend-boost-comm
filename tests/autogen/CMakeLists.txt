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
    $ENV{THIRD_PARTY_DIR}/nlohmannJson/include
)

execute_process(COMMAND python3 ${CMAKE_CURRENT_LIST_DIR}/op_desc_json_auto.py
                                --namespace ${NAMESPACE}
                                --params_path ${OPS_PROJECT_ROOT_DIR}/${CMAKE_PROJECT_NAME}/include)
add_library(mki_test_autogen SHARED ${CMAKE_CURRENT_LIST_DIR}/op_desc_json.cpp)
target_link_libraries(mki_test_autogen PRIVATE simple_ops)
target_compile_definitions(mki_test_autogen PRIVATE OpSpace=${NAMESPACE})

install(TARGETS mki_test_autogen DESTINATION tests)
