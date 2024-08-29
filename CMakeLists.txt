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

cmake_minimum_required(VERSION 3.20)
project("mki")
set(CMAKE_CXX_STANDARD 17)

option(BUILD_TEST_FRAMEWORK "BUILD_TEST_FRAMEWORK" OFF)
option(BUILD_UNIT_TEST "BUILD_UNIT_TEST" OFF)
option(USE_CXX11_ABI "USE_CXX11_ABI" ON)
message(STATUS "BUILD_TEST_FRAMEWORK:${BUILD_TEST_FRAMEWORK}")
message(STATUS "BUILD_UNIT_TEST:${BUILD_UNIT_TEST}")
message(STATUS "USE_CXX11_ABI:${USE_CXX11_ABI}")

include(${PROJECT_SOURCE_DIR}/cmake/host_config.cmake)

include_directories(
    ${PROJECT_SOURCE_DIR}/src/include
    $ENV{ASCEND_HOME_PATH}/include
)

link_directories(
    $ENV{ASCEND_HOME_PATH}/lib64
)

add_subdirectory(src)

if(BUILD_TEST_FRAMEWORK)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/framework)
endif()

if(BUILD_UNIT_TEST)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests/unittest)
endif()

# cmake install stage
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output/mki" CACHE PATH "..." FORCE)
endif()

# cp files
install(FILES ${PROJECT_SOURCE_DIR}/scripts/set_env.sh DESTINATION ./)
file(GLOB_RECURSE SCRIPTS_FILES "${PROJECT_SOURCE_DIR}/scripts/*.py")
list(APPEND SCRIPTS_FILES
    ${PROJECT_SOURCE_DIR}/scripts/build_env_setup.sh
    ${PROJECT_SOURCE_DIR}/scripts/build_ops.sh
)
install(FILES ${SCRIPTS_FILES} DESTINATION ./scripts)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include DESTINATION .)
