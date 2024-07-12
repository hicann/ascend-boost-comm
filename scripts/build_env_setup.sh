#!/bin/bash
# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
