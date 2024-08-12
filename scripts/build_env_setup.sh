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

set -e
function fn_install_lcov()
{
    wget --no-check-certificate https://github.com/linux-test-project/lcov/archive/refs/tags/v1.16.tar.gz
    tar -xvf v1.16.tar.gz
    cd lcov-1.16
    make -j
    make PREFIX=/usr/local/lcov install
}

function fn_install_doxygen()
{
    wget --no-check-certificate https://github.com/doxygen/doxygen/archive/refs/heads/master.tar.gz
    tar -xvf master.tar.gz

    apt-get install flex
    apt-get install bison

    cd doxygen-master
    mkdir build
    cd build
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local/doxygen ../
    make -j
    make install
}

function fn_main()
{
    fn_install_lcov
    fn_install_doxygen
}

fn_main "$@"
