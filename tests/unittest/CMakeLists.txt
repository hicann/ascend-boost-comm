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

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/api TEST_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/operationir TEST_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/utils TEST_SRC)
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/googletest googletest EXCLUDE_FROM_ALL)
add_executable(mki_unittest ${TEST_SRC})
target_include_directories(mki_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/metadef/inc)
target_include_directories(mki_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/include)
target_include_directories(mki_unittest PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_directories(mki_unittest PRIVATE ${PROJECT_SOURCE_DIR}/3rdparty/googletest/lib)
target_link_libraries(mki_unittest PRIVATE mki gtest gtest_main)
target_compile_options(mki_unittest PRIVATE
    -Wno-sign-compare
    -Wno-narrowing
)
target_compile_definitions(mki_unittest PRIVATE fe=MkiFe)

install(TARGETS mki_unittest DESTINATION bin)
